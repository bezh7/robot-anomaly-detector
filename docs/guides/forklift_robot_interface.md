# Forklift Robot Interface

This is the finalized ROS-facing interface for the Isaac Sim Forklift B setup.

## USD

Open the patched USD in Isaac Sim:

```text
/isaac-sim/.local/share/ov/data/forklift_controller/Isaac/Samples/ROS2/Robots/forklift_b_ROS_controller.usd
```

The repository copy is:

```text
sim/isaacsim/forklift_b_ROS_controller.usd
```

The patched USD keeps NVIDIA's original drive and sensor graphs and adds:

```text
/forklift/lift_cmd std_msgs/msg/Float64
```

The lift command is a target lift height in meters. The joint limits are
approximately `-0.15` to `2.0`.

## Final Control Topics

Run the robot interface before path planning:

```bash
infra/isaacsim/run_forklift_robot_interface.sh
```

It starts:

```text
/forklift_cmd  -> normalized Ackermann command input
/ackermann_cmd -> raw Isaac/NVIDIA Ackermann command output
/forklift/odom -> normalized odometry for planning
```

Normalized command convention:

```text
/forklift_cmd.speed > 0          visual forward
/forklift_cmd.steering_angle > 0 visual left
```

Raw Isaac convention remains available for debugging:

```text
/ackermann_cmd.speed > 0          visual reverse
/ackermann_cmd.steering_angle > 0 visual right
```

The calibrated speed scale is:

```text
0.44631542573761773 m/s per raw Isaac speed unit
```

## Odom Convention

Raw `/odom` is physically useful but the forklift model's `base_link +X` points
visually backward.

Use `/forklift/odom` for planning. It keeps the same position as `/odom`, but
rotates orientation by `pi` so `forklift_base_link +X` points visually forward.

Published planning frame:

```text
odom -> forklift_base_link
```

## Smoke Test

With Isaac Sim playing the patched USD, run:

```bash
infra/isaacsim/run_forklift_robot_interface.sh
```

In a second terminal:

```bash
infra/isaacsim/run_forklift_integrated_smoke.sh
```

Expected sequence:

```text
raise lift -> drive straight -> left arc -> stop -> lower lift -> final stop
```

Known good smoke result from May 4, 2026:

```text
straight: 0.25 m/s for 3.98 sim-sec -> 1.019 m
left arc: 0.20 m/s for 5.98 sim-sec -> 1.199 m, 21.5 deg yaw
final stop p95 speed: 0.0058 m/s
```

## Manual Commands

Raise the lift:

```bash
sudo docker exec ros_ws_docker bash -lc '
source /opt/ros/jazzy/setup.bash
source /jazzy_ws/install/setup.bash
export ROS_DOMAIN_ID=0
ros2 topic pub --once /forklift/lift_cmd std_msgs/msg/Float64 "{data: 0.8}"
'
```

Drive forward through the normalized command interface:

```bash
sudo docker exec ros_ws_docker bash -lc '
source /opt/ros/jazzy/setup.bash
source /jazzy_ws/install/setup.bash
export ROS_DOMAIN_ID=0
ros2 topic pub -r 10 /forklift_cmd ackermann_msgs/msg/AckermannDriveStamped \
"{drive: {speed: 0.25, steering_angle: 0.0}}"
'
```

Stop with:

```text
speed: 0.0
steering_angle: 0.0
```
