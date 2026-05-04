#!/usr/bin/env bash
set -euo pipefail

ROS_CONTAINER="${ROS_CONTAINER:-ros_ws_docker}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
REPO_DIR="${RAD_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SHIM_SOURCE="${FORKLIFT_CMD_SHIM_SOURCE:-$REPO_DIR/infra/isaacsim/forklift_cmd_shim.py}"
SHIM_SCRIPT="${FORKLIFT_CMD_SHIM_SCRIPT:-/tmp/forklift_cmd_shim.py}"
LOG_PATH="${FORKLIFT_CMD_SHIM_LOG:-/tmp/forklift_cmd_shim_integrated.csv}"
SCALE="${VISUAL_MPS_PER_RAW_SPEED:-0.44631542573761773}"

if [[ ! -f "$SHIM_SOURCE" && -f /tmp/forklift_cmd_shim.py ]]; then
  SHIM_SOURCE=/tmp/forklift_cmd_shim.py
fi

if [[ ! -f "$SHIM_SOURCE" ]]; then
  echo "Forklift command shim not found: $SHIM_SOURCE" >&2
  echo "Set FORKLIFT_CMD_SHIM_SOURCE or run this script from the repository checkout." >&2
  exit 1
fi

docker cp "$SHIM_SOURCE" "$ROS_CONTAINER:$SHIM_SCRIPT"
docker exec -u root "$ROS_CONTAINER" chmod +x "$SHIM_SCRIPT"

TTY_ARGS=()
if [[ -t 0 ]]; then
  TTY_ARGS=(-it)
fi

docker exec "${TTY_ARGS[@]}" "$ROS_CONTAINER" bash -lc "
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /jazzy_ws/install/setup.bash
export ROS_DOMAIN_ID='$ROS_DOMAIN_ID'
exec python3 '$SHIM_SCRIPT' \
  --visual-mps-per-raw-speed '$SCALE' \
  --log '$LOG_PATH'
"
