#!/usr/bin/env bash
set -euo pipefail

ROS_CONTAINER="${ROS_CONTAINER:-ros_ws_docker}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
REPO_DIR="${RAD_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ODOM_SOURCE="${FORKLIFT_NORMALIZED_ODOM_SOURCE:-$REPO_DIR/infra/isaacsim/forklift_normalized_odom.py}"
ODOM_SCRIPT="${FORKLIFT_NORMALIZED_ODOM_SCRIPT:-/tmp/forklift_normalized_odom.py}"

if [[ ! -f "$ODOM_SOURCE" && -f /tmp/forklift_normalized_odom.py ]]; then
  ODOM_SOURCE=/tmp/forklift_normalized_odom.py
fi

if [[ ! -f "$ODOM_SOURCE" ]]; then
  echo "Forklift normalized odom script not found: $ODOM_SOURCE" >&2
  echo "Set FORKLIFT_NORMALIZED_ODOM_SOURCE or run this script from the repository checkout." >&2
  exit 1
fi

docker cp "$ODOM_SOURCE" "$ROS_CONTAINER:$ODOM_SCRIPT"
docker exec -u root "$ROS_CONTAINER" chmod +x "$ODOM_SCRIPT"

TTY_ARGS=()
if [[ -t 0 ]]; then
  TTY_ARGS=(-it)
fi

docker exec "${TTY_ARGS[@]}" "$ROS_CONTAINER" bash -lc "
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /jazzy_ws/install/setup.bash
export ROS_DOMAIN_ID='$ROS_DOMAIN_ID'
exec python3 '$ODOM_SCRIPT'
"
