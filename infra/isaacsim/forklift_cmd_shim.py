#!/usr/bin/env python3
import argparse
import csv
import math
import time
from pathlib import Path

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class ForkliftCommandShim(Node):
    def __init__(self, args):
        super().__init__("forklift_cmd_shim")
        self.args = args
        self.desired_speed = 0.0
        self.desired_steer = 0.0
        self.desired_acceleration = 0.0
        self.actual_visual_speed = 0.0
        self.integral = 0.0
        self.last_cmd_time = self.get_clock().now()
        self.last_control_time = time.monotonic()
        self.rows = []
        self.start_wall = time.monotonic()

        self.pub = self.create_publisher(AckermannDriveStamped, args.output_topic, 10)
        self.create_subscription(AckermannDriveStamped, args.input_topic, self.on_cmd, 10)
        self.create_subscription(Odometry, args.odom_topic, self.on_odom, 50)
        self.timer = self.create_timer(1.0 / args.rate_hz, self.on_timer)

    def on_cmd(self, msg):
        self.desired_speed = float(msg.drive.speed)
        self.desired_steer = float(msg.drive.steering_angle)
        self.desired_acceleration = float(msg.drive.acceleration)
        self.last_cmd_time = self.get_clock().now()

    def on_odom(self, msg):
        # The forklift model's base_link +X points visually backward. Odom twist
        # is aligned with base_link, so visual-forward speed is -linear.x.
        self.actual_visual_speed = -float(msg.twist.twist.linear.x)

    def clamp(self, value, limit):
        return max(-limit, min(limit, value))

    def on_timer(self):
        now = self.get_clock().now()
        now_wall = time.monotonic()
        dt = max(now_wall - self.last_control_time, 1e-6)
        self.last_control_time = now_wall

        age = (now - self.last_cmd_time).nanoseconds * 1e-9
        desired_speed = self.desired_speed
        desired_steer = self.desired_steer
        if age > self.args.command_timeout:
            desired_speed = 0.0
            desired_steer = 0.0
            desired_acceleration = 0.0
        else:
            desired_acceleration = self.desired_acceleration

        if abs(desired_speed) < self.args.deadband:
            self.integral = 0.0
            feedforward_raw_speed = 0.0
            feedback_raw_speed = 0.0
            raw_speed = 0.0
        else:
            error = desired_speed - self.actual_visual_speed
            feedforward_raw_speed = desired_speed / self.args.visual_mps_per_raw_speed
            feedback_raw_speed = 0.0
            if self.args.feedback == "pi":
                self.integral = self.clamp(self.integral + error * dt, self.args.integral_limit)
                feedback_raw_speed = self.args.kp * error + self.args.ki * self.integral
            else:
                self.integral = 0.0

            raw_speed_unsigned = feedforward_raw_speed + feedback_raw_speed
            raw_speed = self.args.raw_speed_sign * self.clamp(
                raw_speed_unsigned, self.args.max_raw_speed
            )

        raw_steer = self.args.raw_steering_sign * self.clamp(
            desired_steer, self.args.max_steering
        )
        raw_acceleration = self.args.raw_speed_sign * self.clamp(
            desired_acceleration, self.args.max_raw_acceleration
        )

        msg = AckermannDriveStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(raw_speed)
        msg.drive.steering_angle = float(raw_steer)
        msg.drive.acceleration = float(raw_acceleration)
        self.pub.publish(msg)

        if self.args.log:
            self.rows.append(
                {
                    "t_wall": now_wall - self.start_wall,
                    "desired_speed": desired_speed,
                    "actual_visual_speed": self.actual_visual_speed,
                    "speed_error": desired_speed - self.actual_visual_speed,
                    "desired_steer": desired_steer,
                    "desired_acceleration": desired_acceleration,
                    "feedforward_raw_speed": feedforward_raw_speed,
                    "feedback_raw_speed": feedback_raw_speed,
                    "raw_speed": raw_speed,
                    "raw_steer": raw_steer,
                    "raw_acceleration": raw_acceleration,
                    "cmd_age": age,
                }
            )

    def close(self):
        if not self.args.log:
            return
        out = Path(self.args.log)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            fields = [
                "t_wall",
                "desired_speed",
                "actual_visual_speed",
                "speed_error",
                "desired_steer",
                "desired_acceleration",
                "feedforward_raw_speed",
                "feedback_raw_speed",
                "raw_speed",
                "raw_steer",
                "raw_acceleration",
                "cmd_age",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-topic", default="/forklift_cmd")
    parser.add_argument("--output-topic", default="/ackermann_cmd")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument(
        "--visual-mps-per-raw-speed",
        "--speed-scale",
        dest="visual_mps_per_raw_speed",
        type=float,
        default=0.44631542573761773,
        help="Open-loop m/s produced by one raw Isaac speed unit.",
    )
    parser.add_argument(
        "--raw-speed-sign",
        type=float,
        default=-1.0,
        help="Sign converting visual-forward speed to Isaac raw speed.",
    )
    parser.add_argument(
        "--raw-steering-sign",
        type=float,
        default=-1.0,
        help="Sign converting visual-left steering to Isaac raw steering.",
    )
    parser.add_argument("--feedback", choices=["none", "pi"], default="pi")
    parser.add_argument("--kp", type=float, default=0.6)
    parser.add_argument("--ki", type=float, default=0.15)
    parser.add_argument("--integral-limit", type=float, default=1.0)
    parser.add_argument("--max-raw-speed", type=float, default=2.0)
    parser.add_argument("--max-raw-acceleration", type=float, default=3.0)
    parser.add_argument("--max-steering", type=float, default=0.6)
    parser.add_argument("--command-timeout", type=float, default=0.5)
    parser.add_argument("--deadband", type=float, default=1e-3)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--log")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = ForkliftCommandShim(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
