#!/usr/bin/env bash
set -euo pipefail

ROS_CONTAINER="${ROS_CONTAINER:-ros_ws_docker}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
REPO_DIR="${RAD_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SHIM_SOURCE="${FORKLIFT_CMD_SHIM_SOURCE:-$REPO_DIR/infra/isaacsim/forklift_cmd_shim.py}"
ODOM_SOURCE="${FORKLIFT_NORMALIZED_ODOM_SOURCE:-$REPO_DIR/infra/isaacsim/forklift_normalized_odom.py}"
SCALE="${VISUAL_MPS_PER_RAW_SPEED:-0.44631542573761773}"
SHIM_LOG="${FORKLIFT_CMD_SHIM_LOG:-/tmp/forklift_cmd_shim.csv}"

if [[ ! -f "$SHIM_SOURCE" && -f /tmp/forklift_cmd_shim.py ]]; then
  SHIM_SOURCE=/tmp/forklift_cmd_shim.py
fi
if [[ ! -f "$ODOM_SOURCE" && -f /tmp/forklift_normalized_odom.py ]]; then
  ODOM_SOURCE=/tmp/forklift_normalized_odom.py
fi

if [[ ! -f "$SHIM_SOURCE" ]]; then
  echo "Forklift command shim not found: $SHIM_SOURCE" >&2
  exit 1
fi
if [[ ! -f "$ODOM_SOURCE" ]]; then
  echo "Forklift normalized odom script not found: $ODOM_SOURCE" >&2
  exit 1
fi

docker cp "$SHIM_SOURCE" "$ROS_CONTAINER:/tmp/forklift_cmd_shim.py"
docker cp "$ODOM_SOURCE" "$ROS_CONTAINER:/tmp/forklift_normalized_odom.py"
docker exec -u root "$ROS_CONTAINER" chmod +x /tmp/forklift_cmd_shim.py /tmp/forklift_normalized_odom.py

TTY_ARGS=()
if [[ -t 0 ]]; then
  TTY_ARGS=(-it)
fi

docker exec "${TTY_ARGS[@]}" "$ROS_CONTAINER" bash -lc "
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /jazzy_ws/install/setup.bash
export ROS_DOMAIN_ID='$ROS_DOMAIN_ID'

python3 /tmp/forklift_normalized_odom.py &
odom_pid=\$!

cleanup() {
  kill \"\$odom_pid\" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

python3 /tmp/forklift_cmd_shim.py \
  --visual-mps-per-raw-speed '$SCALE' \
  --log '$SHIM_LOG'
"
