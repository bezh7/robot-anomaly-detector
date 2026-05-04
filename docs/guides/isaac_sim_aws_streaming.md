# Isaac Sim AWS Streaming Setup

This documents the Isaac Sim cloud streaming setup that was built manually on
AWS after the repo instructions did not work.

Use this as the operator record for the current instance and as the repeatable
setup flow for a replacement instance.

## Current Instance

Only this EC2 instance was created and modified for the Isaac Sim setup:

```text
AWS profile:       codex
AWS region:        us-east-1
Instance ID:       i-087c0015e521bd90f
Current type:      g5.2xlarge
Original type:     g6e.2xlarge
Public IP:         changes on stop/start; check AWS before connecting
Key file:          ~/.ssh/rad-isaac-key.pem
Security group:    sg-04f5a286f2d19c94c
Remote stack dir:  /opt/isaacsim-stream
```

The instance was originally launched as `g6e.2xlarge`, but AWS returned
`InsufficientInstanceCapacity` when it was restarted. It was changed to
`g5.2xlarge` so the same disk and setup could come back online.

The public IP changes after stop/start unless an Elastic IP is attached. When
that happens, update `/opt/isaacsim-stream/.env` and rebuild the web viewer.

## Network Rules

The security group is intentionally narrow and only allows the current client IP
to connect.

Required inbound rules from the client IP:

```text
TCP 22      SSH
TCP 8210    browser web viewer
TCP 49100   Isaac Sim WebRTC signaling
UDP 47998   Isaac Sim media stream
```

Check the current client IP:

```bash
curl -s https://checkip.amazonaws.com
```

Do not modify unrelated EC2 instances in this AWS account.

## NVIDIA Version Choice

Use Isaac Sim `5.1.0` for this setup.

`6.0.0-dev2` streamed, but the asset browser was unreliable because it pointed
at 6.0/staging-style asset paths. NVIDIA's published downloadable Isaac Sim
asset packs and the stable cloud asset tree are currently aligned with `5.1.0`.

Container image in use:

```text
nvcr.io/nvidia/isaac-sim:5.1.0
```

NVIDIA's public ECR tag was not available during setup. The working image came
from NVIDIA NGC (`nvcr.io`).

Reference docs:

- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html
- https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_advanced_cloud_setup_aws.html
- https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_ros.html
- https://github.com/isaac-sim/IsaacSim/tree/develop/tools/docker

## Host Setup

The instance uses an NVIDIA Deep Learning Base AMI:

```text
AMI: ami-0b479b3c80d39efa7
OS:  Ubuntu 22.04
Root volume: 250 GB gp3
Ephemeral NVMe: /opt/dlami/nvme
```

Docker, Docker Compose, and NVIDIA Container Toolkit were installed on the host.
Useful checks:

```bash
nvidia-smi
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Remote Directory Layout

The working setup lives under:

```text
/opt/isaacsim-stream
```

Important files:

```text
/opt/isaacsim-stream/.env
/opt/isaacsim-stream/tools/docker/docker-compose.yml
/opt/isaacsim-stream/docker-compose.ros2.yml
/opt/isaacsim-stream/docker-compose.assets.yml
/opt/isaacsim-stream/local-assets.kit
```

The base `tools/docker/docker-compose.yml` came from NVIDIA's IsaacSim repo.
Always include the ROS and asset override files when starting the stack.

## Environment File

Current `/opt/isaacsim-stream/.env` shape:

```bash
ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
ISAACSIM_HOST=PUBLIC_IP
ISAACSIM_SIGNAL_PORT=49100
ISAACSIM_STREAM_PORT=47998
WEB_VIEWER_PORT=8210
ISAAC_SIM_DATA=/home/ubuntu/docker/isaac-sim
GPU_DEVICE=all
ROS_DOMAIN_ID=0
```

If the EC2 public IP changes:

```bash
cd /opt/isaacsim-stream
sudo sed -i 's/^ISAACSIM_HOST=.*/ISAACSIM_HOST=NEW_PUBLIC_IP/' .env
```

Then rebuild/recreate the stack so the web viewer JavaScript uses the new IP.

## Start The Stack

Use this exact compose command:

```bash
cd /opt/isaacsim-stream

sudo docker compose --env-file .env -p isim \
  -f tools/docker/docker-compose.yml \
  -f docker-compose.ros2.yml \
  -f docker-compose.assets.yml \
  up -d --build
```

Check status:

```bash
sudo docker compose --env-file .env -p isim \
  -f tools/docker/docker-compose.yml \
  -f docker-compose.ros2.yml \
  -f docker-compose.assets.yml \
  ps
```

Expected containers:

```text
isim-isaac-sim-1    nvcr.io/nvidia/isaac-sim:5.1.0   healthy
isim-web-viewer-1   isim-web-viewer                  healthy
ros_ws_docker       osrf/ros:jazzy-desktop           running
```

Open the browser viewer:

```text
http://PUBLIC_IP:8210
```

## Stop The Stack Or Instance

Stop only the compose stack:

```bash
cd /opt/isaacsim-stream

sudo docker compose --env-file .env -p isim \
  -f tools/docker/docker-compose.yml \
  -f docker-compose.ros2.yml \
  -f docker-compose.assets.yml \
  down
```

Stop the EC2 instance without terminating it:

```bash
aws ec2 stop-instances \
  --profile codex \
  --region us-east-1 \
  --instance-ids i-087c0015e521bd90f
```

Wait for stopped:

```bash
aws ec2 wait instance-stopped \
  --profile codex \
  --region us-east-1 \
  --instance-ids i-087c0015e521bd90f
```

## Asset Setup

The full complete asset archive was not a good fit for the original root disk.
It is about 150 GB compressed before extraction:

```text
isaac-sim-assets-complete-5.1.0.001.zip   50 GiB
isaac-sim-assets-complete-5.1.0.002.zip   50 GiB
isaac-sim-assets-complete-5.1.0.003.zip   40 GiB
```

Instead, the smaller documented category packs were downloaded:

```text
isaac-sim-assets-robots_and_sensors-5.1.0.zip
isaac-sim-assets-materials_and_props-5.1.0.zip
isaac-sim-assets-environments-5.1.0.zip
```

Temporary download directory:

```text
/opt/dlami/nvme/isaac-assets-downloads
```

Persistent extracted asset root:

```text
/home/ubuntu/isaacsim_assets/Assets/Isaac/5.1
```

Container-visible asset root:

```text
/isaacsim_assets/Assets/Isaac/5.1
```

Installed categories:

```text
Environments
Materials
Props
Robots
Sensors
```

Download and extract commands:

```bash
sudo apt-get update
sudo apt-get install -y aria2 unzip

sudo mkdir -p /opt/dlami/nvme/isaac-assets-downloads /home/ubuntu/isaacsim_assets
sudo chown -R ubuntu:ubuntu /opt/dlami/nvme/isaac-assets-downloads /home/ubuntu/isaacsim_assets

cd /opt/dlami/nvme/isaac-assets-downloads

aria2c --continue=true --max-connection-per-server=16 --split=16 \
  --min-split-size=64M --file-allocation=none \
  https://downloads.isaacsim.nvidia.com/isaac-sim-assets-robots_and_sensors-5.1.0.zip

aria2c --continue=true --max-connection-per-server=16 --split=16 \
  --min-split-size=64M --file-allocation=none \
  https://downloads.isaacsim.nvidia.com/isaac-sim-assets-materials_and_props-5.1.0.zip

aria2c --continue=true --max-connection-per-server=16 --split=16 \
  --min-split-size=64M --file-allocation=none \
  https://downloads.isaacsim.nvidia.com/isaac-sim-assets-environments-5.1.0.zip

for zip in isaac-sim-assets-*-5.1.0.zip; do
  unzip -q -n "$zip" -d /home/ubuntu/isaacsim_assets
done

sudo chown -R 1234:1234 /home/ubuntu/isaacsim_assets
```

Do not mount the assets inside the container as
`/home/ubuntu/isaacsim_assets`. The Isaac Sim container runs as uid/gid `1234`
and cannot necessarily traverse `/home/ubuntu`. Mount the host directory to a
simple container path such as `/isaacsim_assets`.

Current `/opt/isaacsim-stream/docker-compose.assets.yml`:

```yaml
services:
  isaac-sim:
    volumes:
      - /opt/isaacsim-stream/local-assets.kit:/isaac-sim/local-assets.kit:ro
      - /home/ubuntu/isaacsim_assets:/isaacsim_assets:ro
    command:
      - --merge-config=/isaac-sim/local-assets.kit
      - --/persistent/isaac/asset_root/default=/isaacsim_assets/Assets/Isaac/5.1
```

Current `/opt/isaacsim-stream/local-assets.kit`:

```toml
[settings]
persistent.isaac.asset_root.default = "/isaacsim_assets/Assets/Isaac/5.1"

exts."isaacsim.gui.content_browser".folders = [
    "/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots",
    "/isaacsim_assets/Assets/Isaac/5.1/Isaac/Props",
    "/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments",
    "/isaacsim_assets/Assets/Isaac/5.1/Isaac/Materials",
    "/isaacsim_assets/Assets/Isaac/5.1/Isaac/Sensors",
]

exts."isaacsim.asset.browser".folders = [
    "/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots",
    "/isaacsim_assets/Assets/Isaac/5.1/Isaac/Props",
    "/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments",
    "/isaacsim_assets/Assets/Isaac/5.1/Isaac/Materials",
    "/isaacsim_assets/Assets/Isaac/5.1/Isaac/Sensors",
]
```

Verification:

```bash
sudo docker exec isim-isaac-sim-1 bash -lc \
  'ls -ld /isaacsim_assets/Assets/Isaac/5.1/Isaac/{Environments,Materials,Props,Robots,Sensors}'
```

Expected examples:

```text
/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse
/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Hospital
/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots
/isaacsim_assets/Assets/Isaac/5.1/Isaac/Sensors
```

## ROS 2 Setup

NVIDIA's ROS workspaces repo was cloned on the instance:

```text
/home/ubuntu/IsaacSim-ros_workspaces
```

Workspace in use:

```text
/home/ubuntu/IsaacSim-ros_workspaces/jazzy_ws
```

The ROS sidecar container:

```bash
sudo docker run -d --net=host \
  --env ROS_DOMAIN_ID=0 \
  --env FASTRTPS_DEFAULT_PROFILES_FILE=/jazzy_ws/fastdds.xml \
  --env DEBIAN_FRONTEND=noninteractive \
  -v /home/ubuntu/IsaacSim-ros_workspaces/jazzy_ws:/jazzy_ws \
  --name ros_ws_docker \
  osrf/ros:jazzy-desktop sleep infinity
```

Build command used:

```bash
sudo docker exec ros_ws_docker bash -lc '
set -eo pipefail
cd /jazzy_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro=jazzy -y
colcon build --symlink-install
'
```

The Isaac Sim container receives the same ROS domain and Fast DDS profile via
`/opt/isaacsim-stream/docker-compose.ros2.yml`:

```yaml
services:
  isaac-sim:
    environment:
      - FASTRTPS_DEFAULT_PROFILES_FILE=/isaac-sim/fastdds.xml
      - ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
    volumes:
      - /home/ubuntu/IsaacSim-ros_workspaces/jazzy_ws/fastdds.xml:/isaac-sim/fastdds.xml:ro
```

Verify ROS packages:

```bash
sudo docker exec ros_ws_docker bash -lc '
source /opt/ros/jazzy/setup.bash
source /jazzy_ws/install/setup.bash
ros2 pkg list | grep -E "^(carter_navigation|isaac_ros_navigation_goal|cmdvel_to_ackermann|isaacsim)$" | sort
'
```

Expected output:

```text
carter_navigation
cmdvel_to_ackermann
isaac_ros_navigation_goal
isaacsim
```

`ros2 topic list` only showing `/parameter_events` and `/rosout` is normal
until a scene/graph in Isaac Sim starts publishing topics.

## Web Viewer Versus Native WebRTC Client

The browser web viewer works and is the recommended path for this instance:

```text
http://PUBLIC_IP:8210
```

The native Isaac Sim WebRTC Streaming app did not connect reliably even after
closing the browser viewer. Server-side checks showed the expected listeners:

```text
TCP 49100   kit signaling
UDP 47998   kit media
TCP 8210    node web viewer
```

The browser viewer works because its built JavaScript hardcodes the current
public IP, signaling port, and media port. The native app appears more sensitive
to the endpoint advertised by Isaac Sim and to stale WebRTC sessions. Since the
web viewer streams the same Isaac Sim process, it is acceptable to use the web
viewer for now.

## Troubleshooting

Check EC2 state:

```bash
aws ec2 describe-instances \
  --profile codex \
  --region us-east-1 \
  --instance-ids i-087c0015e521bd90f \
  --query 'Reservations[0].Instances[0].{State:State.Name,Type:InstanceType,PublicIp:PublicIpAddress}' \
  --output table
```

Check ports from the client machine:

```bash
curl -I --max-time 10 http://PUBLIC_IP:8210/
nc -vz -w 5 PUBLIC_IP 49100
```

Check listeners on the instance:

```bash
sudo ss -lntup | grep -E ':(8210|49100|47998)'
```

Check logs:

```bash
sudo docker logs --tail 200 isim-isaac-sim-1
sudo docker logs --tail 100 isim-web-viewer-1
```

Useful log signals:

```text
Isaac Sim Full Streaming Version: 5.1.0-rc.19
Isaac Sim Full Streaming App is loaded.
isaacsim.ros2.bridge startup
```

Asset browser thumbnail warnings for cloud URLs are noisy but not necessarily
fatal. Verify local assets by checking `/isaacsim_assets` inside the container.

If AWS returns `InsufficientInstanceCapacity` when starting `g6e.2xlarge`, either
retry later or stop the instance and change only this instance to an available
GPU type such as `g5.2xlarge`:

```bash
aws ec2 modify-instance-attribute \
  --profile codex \
  --region us-east-1 \
  --instance-id i-087c0015e521bd90f \
  --instance-type '{"Value":"g5.2xlarge"}'
```

Then start it:

```bash
aws ec2 start-instances \
  --profile codex \
  --region us-east-1 \
  --instance-ids i-087c0015e521bd90f
```

Remember to update `ISAACSIM_HOST` after any stop/start that changes the public
IP.
