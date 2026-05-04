#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64


def yaw_from_quat(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def unwrap(angles):
    if not angles:
        return []
    out = [angles[0]]
    offset = 0.0
    prev = angles[0]
    for angle in angles[1:]:
        delta = angle - prev
        if delta > math.pi:
            offset -= 2.0 * math.pi
        elif delta < -math.pi:
            offset += 2.0 * math.pi
        out.append(angle + offset)
        prev = angle
    return out


def mean(values):
    values = [v for v in values if math.isfinite(v)]
    return statistics.fmean(values) if values else 0.0


def percentile(values, pct):
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return 0.0
    idx = (len(values) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-topic", default="/forklift_cmd")
    parser.add_argument("--lift-topic", default="/forklift/lift_cmd")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--clock-topic", default="/clock")
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--steer", type=float, default=0.45)
    parser.add_argument("--lift-low", type=float, default=0.0)
    parser.add_argument("--lift-high", type=float, default=0.8)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--wait-timeout", type=float, default=30.0)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    prefix = Path(args.output_prefix or f"/tmp/forklift_integrated_smoke_{stamp}")

    rclpy.init()
    node = rclpy.create_node("forklift_integrated_smoke")
    drive_pub = node.create_publisher(AckermannDriveStamped, args.drive_topic, 10)
    lift_pub = node.create_publisher(Float64, args.lift_topic, 10)

    rows = []
    clock_state = {"t_clock": 0.0}
    odom_state = {"seen": False}
    state = {"phase": "startup", "speed": 0.0, "steer": 0.0, "lift": args.lift_low}
    start_wall = time.monotonic()

    def clock_cb(msg):
        clock_state["t_clock"] = msg.clock.sec + msg.clock.nanosec * 1e-9

    def odom_cb(msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        odom_state["seen"] = True
        rows.append(
            {
                "t_wall": time.monotonic() - start_wall,
                "t_clock": clock_state["t_clock"],
                "phase": state["phase"],
                "cmd_speed": state["speed"],
                "cmd_steer": state["steer"],
                "cmd_lift": state["lift"],
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "yaw": yaw_from_quat(q.x, q.y, q.z, q.w),
                "vx": lin.x,
                "vy": lin.y,
                "vz": lin.z,
                "wx": ang.x,
                "wy": ang.y,
                "wz": ang.z,
            }
        )

    node.create_subscription(Odometry, args.odom_topic, odom_cb, 50)
    node.create_subscription(Clock, args.clock_topic, clock_cb, 50)

    def publish(speed, steer, lift):
        drive = AckermannDriveStamped()
        drive.header.stamp = node.get_clock().now().to_msg()
        drive.header.frame_id = "base_link"
        drive.drive.speed = float(speed)
        drive.drive.steering_angle = float(steer)
        drive_pub.publish(drive)

        lift_msg = Float64()
        lift_msg.data = float(lift)
        lift_pub.publish(lift_msg)

    def sim_time_available():
        return clock_state["t_clock"] > 0.0

    def current_phase_time():
        if sim_time_available():
            return clock_state["t_clock"]
        return time.monotonic()

    def wait_for_ready():
        deadline = time.monotonic() + args.wait_timeout
        while time.monotonic() < deadline:
            publish(0.0, 0.0, args.lift_low)
            rclpy.spin_once(node, timeout_sec=0.1)
            drive_subs = drive_pub.get_subscription_count()
            lift_subs = lift_pub.get_subscription_count()
            if odom_state["seen"] and sim_time_available() and drive_subs > 0 and lift_subs > 0:
                return {
                    "drive_subscription_count": drive_subs,
                    "lift_subscription_count": lift_subs,
                    "clock_seen": True,
                }
        raise SystemExit(
            "Timed out waiting for /odom, /forklift_cmd subscriber, and /forklift/lift_cmd subscriber. "
            f"seen_odom={odom_state['seen']} drive_subs={drive_pub.get_subscription_count()} "
            f"lift_subs={lift_pub.get_subscription_count()} clock={clock_state['t_clock']}"
        )

    readiness = wait_for_ready()

    def run_phase(name, duration, speed, steer, lift):
        state["phase"] = name
        state["speed"] = float(speed)
        state["steer"] = float(steer)
        state["lift"] = float(lift)
        start_phase_time = current_phase_time()
        deadline = start_phase_time + duration
        wall_deadline = time.monotonic() + max(90.0, duration * 60.0)
        period = 1.0 / args.rate_hz
        print(
            f"phase={name} duration={duration:.2f}s speed={speed:.3f} "
            f"steer={steer:.3f} lift={lift:.3f}",
            flush=True,
        )
        while current_phase_time() < deadline:
            if time.monotonic() > wall_deadline:
                raise SystemExit(
                    f"Timed out during phase {name}: phase_time={current_phase_time():.3f}, "
                    f"target={deadline:.3f}"
                )
            publish(speed, steer, lift)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
            rclpy.spin_once(node, timeout_sec=0.0)
        print(f"phase={name} done", flush=True)

    run_phase("pre_stop", 2.0, 0.0, 0.0, args.lift_low)
    run_phase("lift_raise", 2.0, 0.0, 0.0, args.lift_high)
    run_phase("straight_forward", 4.0, args.speed, 0.0, args.lift_high)
    run_phase("left_arc", 6.0, args.speed * 0.8, args.steer, args.lift_high)
    run_phase("stop_high", 1.5, 0.0, 0.0, args.lift_high)
    run_phase("lift_lower", 2.0, 0.0, 0.0, args.lift_low)
    run_phase("final_stop", 2.0, 0.0, 0.0, args.lift_low)

    for _ in range(20):
        publish(0.0, 0.0, args.lift_low)
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.05)

    csv_path = prefix.with_suffix(".csv")
    summary_path = prefix.with_suffix(".summary.json")
    fields = [
        "t_wall",
        "t_clock",
        "phase",
        "cmd_speed",
        "cmd_steer",
        "cmd_lift",
        "x",
        "y",
        "z",
        "yaw",
        "vx",
        "vy",
        "vz",
        "wx",
        "wy",
        "wz",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    use_clock = any(float(row["t_clock"]) > 0.0 for row in rows)
    time_key = "t_clock" if use_clock else "t_wall"
    yaws = unwrap([float(row["yaw"]) for row in rows])
    for row, yaw in zip(rows, yaws):
        row["_yaw_unwrapped"] = yaw

    moving_rows = [row for row in rows if abs(float(row["cmd_speed"])) > 1e-6]
    final_rows = [row for row in rows if row["phase"] == "final_stop"]
    arc_rows = [row for row in rows if row["phase"] == "left_arc"]

    def displacement(sample_rows):
        if len(sample_rows) < 2:
            return 0.0
        dx = float(sample_rows[-1]["x"]) - float(sample_rows[0]["x"])
        dy = float(sample_rows[-1]["y"]) - float(sample_rows[0]["y"])
        return math.hypot(dx, dy)

    summary = {
        "csv": str(csv_path),
        "rows": len(rows),
        "time_source": time_key,
        "readiness": readiness,
        "commanded_speed_mps": args.speed,
        "commanded_left_steer_rad": args.steer,
        "lift_low_m": args.lift_low,
        "lift_high_m": args.lift_high,
        "moving_displacement_m": displacement(moving_rows),
        "straight_displacement_m": displacement([row for row in rows if row["phase"] == "straight_forward"]),
        "arc_displacement_m": displacement(arc_rows),
        "arc_yaw_change_deg": math.degrees(float(arc_rows[-1]["_yaw_unwrapped"]) - float(arc_rows[0]["_yaw_unwrapped"]))
        if len(arc_rows) >= 2
        else 0.0,
        "moving_visual_twist_speed_mean_mps": mean([-float(row["vx"]) for row in moving_rows]),
        "moving_twist_norm_speed_p95_mps": percentile(
            [math.hypot(float(row["vx"]), float(row["vy"])) for row in moving_rows], 0.95
        ),
        "final_stop_twist_norm_mean_mps": mean(
            [math.hypot(float(row["vx"]), float(row["vy"])) for row in final_rows]
        ),
        "final_stop_twist_norm_p95_mps": percentile(
            [math.hypot(float(row["vx"]), float(row["vy"])) for row in final_rows], 0.95
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
