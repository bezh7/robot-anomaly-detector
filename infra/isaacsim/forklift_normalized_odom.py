#!/usr/bin/env python3
import argparse
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw):
    half = 0.5 * yaw
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(half),
        "w": math.cos(half),
    }


class ForkliftNormalizedOdom(Node):
    def __init__(self, args):
        super().__init__("forklift_normalized_odom")
        self.args = args
        self.publisher = self.create_publisher(Odometry, args.output_topic, 20)
        self.tf_broadcaster = TransformBroadcaster(self) if args.publish_tf else None
        self.subscription = self.create_subscription(Odometry, args.input_topic, self.on_odom, 50)
        self.get_logger().info(
            f"Publishing normalized forklift odom {args.output_topic}: "
            f"{args.parent_frame}->{args.child_frame} from {args.input_topic}"
        )

    def on_odom(self, msg):
        out = Odometry()
        out.header = msg.header
        out.header.frame_id = self.args.parent_frame
        out.child_frame_id = self.args.child_frame
        out.pose = msg.pose

        yaw = yaw_from_quat(msg.pose.pose.orientation) + math.pi
        q = quat_from_yaw(yaw)
        out.pose.pose.orientation.x = q["x"]
        out.pose.pose.orientation.y = q["y"]
        out.pose.pose.orientation.z = q["z"]
        out.pose.pose.orientation.w = q["w"]

        out.twist = msg.twist
        out.twist.twist.linear.x = -msg.twist.twist.linear.x
        out.twist.twist.linear.y = -msg.twist.twist.linear.y
        out.twist.twist.linear.z = msg.twist.twist.linear.z
        out.twist.twist.angular = msg.twist.twist.angular

        self.publisher.publish(out)

        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header = out.header
            transform.child_frame_id = out.child_frame_id
            transform.transform.translation.x = out.pose.pose.position.x
            transform.transform.translation.y = out.pose.pose.position.y
            transform.transform.translation.z = out.pose.pose.position.z
            transform.transform.rotation = out.pose.pose.orientation
            self.tf_broadcaster.sendTransform(transform)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-topic", default="/odom")
    parser.add_argument("--output-topic", default="/forklift/odom")
    parser.add_argument("--parent-frame", default="odom")
    parser.add_argument("--child-frame", default="forklift_base_link")
    parser.add_argument("--publish-tf", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = ForkliftNormalizedOdom(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
