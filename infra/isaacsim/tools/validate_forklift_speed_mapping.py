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


def yaw_from_quat(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


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


def parse_csv_floats(value):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def linear_fit(xs, ys):
    if len(xs) < 2:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0, "through_origin_slope": 0.0}

    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom if denom else 0.0
    intercept = y_mean - slope * x_mean
    pred = [slope * x + intercept for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    origin_denom = sum(x * x for x in xs)
    through_origin_slope = sum(x * y for x, y in zip(xs, ys)) / origin_denom if origin_denom else 0.0
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "through_origin_slope": through_origin_slope,
    }


def scale(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def polyline(points, x_min, x_max, y_min, y_max, x0, y0, w, h):
    out = []
    for x, y in points:
        out.append(f"{scale(x, x_min, x_max, x0, x0 + w):.2f},{scale(y, y_min, y_max, y0 + h, y0):.2f}")
    return " ".join(out)


def make_svg(path, results, fit):
    if not results:
        return

    xs = [row["raw_speed_magnitude"] for row in results]
    ys = [row["pose_speed_mean_mps"] for row in results]
    residuals = [row["pose_speed_mean_mps"] - row["fit_pose_speed_mps"] for row in results]

    width = 980
    height = 520
    x0, y0, w, h = 80, 70, 390, 310
    x1, y1, w1, h1 = 560, 70, 340, 310

    x_min = min(xs) * 0.9
    x_max = max(xs) * 1.08
    y_min = 0.0
    y_max = max(max(ys), max(row["fit_pose_speed_mps"] for row in results)) * 1.12
    r_abs = max(max(abs(r) for r in residuals), 1e-6)

    fit_points = [
        (x_min, fit["slope"] * x_min + fit["intercept"]),
        (x_max, fit["slope"] * x_max + fit["intercept"]),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f7f9"/>',
        '<text x="70" y="34" font-size="20" font-weight="700">Forklift raw speed mapping</text>',
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="white" stroke="#c8ccd2"/>',
        f'<text x="{x0}" y="{y0 - 10}" font-size="14" font-weight="600">Actual speed vs raw command</text>',
        f'<text x="{x0 + w / 2}" y="{y0 + h + 36}" font-size="12" text-anchor="middle">raw speed magnitude</text>',
        f'<text x="{x0 - 45}" y="{y0 + h / 2}" font-size="12" text-anchor="middle" transform="rotate(-90 {x0 - 45} {y0 + h / 2})">m/s from pose</text>',
        f'<polyline points="{polyline(fit_points, x_min, x_max, y_min, y_max, x0, y0, w, h)}" fill="none" stroke="#175cd3" stroke-width="2"/>',
        f'<rect x="{x1}" y="{y1}" width="{w1}" height="{h1}" fill="white" stroke="#c8ccd2"/>',
        f'<text x="{x1}" y="{y1 - 10}" font-size="14" font-weight="600">Fit residuals</text>',
        f'<text x="{x1 + w1 / 2}" y="{y1 + h1 + 36}" font-size="12" text-anchor="middle">raw speed magnitude</text>',
        f'<text x="{x1 - 42}" y="{y1 + h1 / 2}" font-size="12" text-anchor="middle" transform="rotate(-90 {x1 - 42} {y1 + h1 / 2})">m/s</text>',
    ]

    zero_y = scale(0.0, -r_abs, r_abs, y1 + h1, y1)
    parts.append(f'<line x1="{x1}" y1="{zero_y:.2f}" x2="{x1 + w1}" y2="{zero_y:.2f}" stroke="#98a2b3" stroke-dasharray="5,5"/>')

    for row, residual in zip(results, residuals):
        cx = scale(row["raw_speed_magnitude"], x_min, x_max, x0, x0 + w)
        cy = scale(row["pose_speed_mean_mps"], y_min, y_max, y0 + h, y0)
        parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5" fill="#b42318"/>')

        rx = scale(row["raw_speed_magnitude"], x_min, x_max, x1, x1 + w1)
        ry = scale(residual, -r_abs, r_abs, y1 + h1, y1)
        parts.append(f'<circle cx="{rx:.2f}" cy="{ry:.2f}" r="5" fill="#7a5af8"/>')

    parts.extend(
        [
            f'<text x="80" y="450" font-size="13">fit: actual_mps = {fit["slope"]:.4f} * raw + {fit["intercept"]:.4f}</text>',
            f'<text x="80" y="472" font-size="13">R2 = {fit["r2"]:.4f}; through-origin scale = {fit["through_origin_slope"]:.4f} m/s per raw unit</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-topic", default="/ackermann_cmd")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--clock-topic", default="/clock")
    parser.add_argument("--raw-speed-magnitudes", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--raw-speed-sign", type=float, default=-1.0)
    parser.add_argument("--steer", type=float, default=0.0)
    parser.add_argument("--pre-stop-seconds", type=float, default=2.0)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--measure-seconds", type=float, default=4.0)
    parser.add_argument("--between-stop-seconds", type=float, default=1.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--odom-timeout", type=float, default=20.0)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    prefix = Path(args.output_prefix or f"/tmp/forklift_speed_mapping_{stamp}")
    raw_magnitudes = parse_csv_floats(args.raw_speed_magnitudes)

    rclpy.init()
    node = rclpy.create_node("validate_forklift_speed_mapping")
    pub = node.create_publisher(AckermannDriveStamped, args.command_topic, 10)

    rows = []
    state = {
        "phase": "startup",
        "raw_speed": 0.0,
        "raw_speed_magnitude": 0.0,
    }
    clock_state = {"t_clock": 0.0}
    start_wall = time.monotonic()
    last_odom = {"seen": False}

    def clock_cb(msg):
        clock_state["t_clock"] = msg.clock.sec + msg.clock.nanosec * 1e-9

    def odom_cb(msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        last_odom["seen"] = True
        rows.append(
            {
                "t_wall": time.monotonic() - start_wall,
                "t_clock": clock_state["t_clock"],
                "phase": state["phase"],
                "raw_speed": state["raw_speed"],
                "raw_speed_magnitude": state["raw_speed_magnitude"],
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

    def publish(raw_speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(raw_speed)
        msg.drive.steering_angle = float(args.steer)
        pub.publish(msg)

    deadline = time.monotonic() + args.odom_timeout
    while time.monotonic() < deadline and not last_odom["seen"]:
        publish(0.0)
        rclpy.spin_once(node, timeout_sec=0.1)

    if not last_odom["seen"]:
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(f"Timed out waiting for odometry on {args.odom_topic}")

    def run_phase(name, duration, raw_speed, raw_speed_magnitude):
        state["phase"] = name
        state["raw_speed"] = float(raw_speed)
        state["raw_speed_magnitude"] = float(raw_speed_magnitude)
        deadline = time.monotonic() + duration
        period = 1.0 / args.rate_hz
        while time.monotonic() < deadline:
            publish(raw_speed)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
            rclpy.spin_once(node, timeout_sec=0.0)

    run_phase("pre_stop", args.pre_stop_seconds, 0.0, 0.0)
    for magnitude in raw_magnitudes:
        raw_speed = args.raw_speed_sign * magnitude
        run_phase(f"settle_raw_{magnitude:g}", args.settle_seconds, raw_speed, magnitude)
        run_phase(f"measure_raw_{magnitude:g}", args.measure_seconds, raw_speed, magnitude)
        run_phase(f"stop_after_{magnitude:g}", args.between_stop_seconds, 0.0, 0.0)

    for _ in range(20):
        publish(0.0)
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.05)

    csv_path = prefix.with_suffix(".csv")
    results_path = prefix.with_suffix(".results.csv")
    summary_path = prefix.with_suffix(".summary.json")
    svg_path = prefix.with_suffix(".svg")

    fields = [
        "t_wall",
        "t_clock",
        "phase",
        "raw_speed",
        "raw_speed_magnitude",
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

    results = []
    for magnitude in raw_magnitudes:
        phase = f"measure_raw_{magnitude:g}"
        phase_rows = [row for row in rows if row["phase"] == phase]
        if len(phase_rows) < 3:
            continue

        t0 = float(phase_rows[0][time_key])
        t1 = float(phase_rows[-1][time_key])
        dt = max(t1 - t0, 1e-9)
        x0 = float(phase_rows[0]["x"])
        y0 = float(phase_rows[0]["y"])
        x1 = float(phase_rows[-1]["x"])
        y1 = float(phase_rows[-1]["y"])
        dx = x1 - x0
        dy = y1 - y0
        displacement = math.hypot(dx, dy)
        chord_speed = displacement / dt

        pose_speeds = []
        for prev, cur in zip(phase_rows, phase_rows[1:]):
            prev_t = float(prev[time_key])
            cur_t = float(cur[time_key])
            step_dt = cur_t - prev_t
            if step_dt <= 1e-9:
                continue
            step_dist = math.hypot(float(cur["x"]) - float(prev["x"]), float(cur["y"]) - float(prev["y"]))
            pose_speeds.append(step_dist / step_dt)

        if displacement > 1e-9:
            ux = dx / displacement
            uy = dy / displacement
            lateral = [
                -(float(row["x"]) - x0) * uy + (float(row["y"]) - y0) * ux
                for row in phase_rows
            ]
        else:
            lateral = [0.0]

        result = {
            "raw_speed_magnitude": magnitude,
            "raw_speed": args.raw_speed_sign * magnitude,
            "sample_count": len(phase_rows),
            "duration_s": dt,
            "displacement_m": displacement,
            "chord_speed_mps": chord_speed,
            "pose_speed_mean_mps": mean(pose_speeds),
            "pose_speed_p50_mps": percentile(pose_speeds, 0.5),
            "pose_speed_p95_mps": percentile(pose_speeds, 0.95),
            "twist_visual_speed_mean_mps": mean([-float(row["vx"]) for row in phase_rows]),
            "twist_norm_speed_mean_mps": mean(
                [math.hypot(float(row["vx"]), float(row["vy"])) for row in phase_rows]
            ),
            "twist_vx_mean": mean([float(row["vx"]) for row in phase_rows]),
            "twist_vy_mean": mean([float(row["vy"]) for row in phase_rows]),
            "max_abs_lateral_deviation_m": max(abs(v) for v in lateral),
            "yaw_change_deg": math.degrees(float(phase_rows[-1]["yaw"]) - float(phase_rows[0]["yaw"])),
        }
        result["mps_per_raw_pose"] = result["pose_speed_mean_mps"] / max(magnitude, 1e-9)
        result["mps_per_raw_twist"] = result["twist_visual_speed_mean_mps"] / max(magnitude, 1e-9)
        results.append(result)

    fit = linear_fit(
        [row["raw_speed_magnitude"] for row in results],
        [row["pose_speed_mean_mps"] for row in results],
    )
    for row in results:
        row["fit_pose_speed_mps"] = fit["slope"] * row["raw_speed_magnitude"] + fit["intercept"]
        row["fit_residual_mps"] = row["pose_speed_mean_mps"] - row["fit_pose_speed_mps"]

    if results:
        with results_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "csv": str(csv_path),
        "results_csv": str(results_path),
        "plot": str(svg_path),
        "time_source": time_key,
        "raw_speed_sign": args.raw_speed_sign,
        "fit_pose_speed_mps_per_raw": fit,
        "recommended_visual_mps_per_raw_speed": fit["through_origin_slope"],
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    make_svg(svg_path, results, fit)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"csv={csv_path}")
    print(f"results={results_path}")
    print(f"summary={summary_path}")
    print(f"plot={svg_path}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
