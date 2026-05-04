#!/usr/bin/env bash
set -euo pipefail

ROS_CONTAINER="${ROS_CONTAINER:-ros_ws_docker}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
REPO_DIR="${RAD_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SMOKE_SOURCE="${FORKLIFT_INTEGRATED_SMOKE_SOURCE:-$REPO_DIR/infra/isaacsim/forklift_integrated_smoke.py}"
SMOKE_SCRIPT="${FORKLIFT_INTEGRATED_SMOKE_SCRIPT:-/tmp/forklift_integrated_smoke.py}"

if [[ ! -f "$SMOKE_SOURCE" && -f /tmp/forklift_integrated_smoke.py ]]; then
  SMOKE_SOURCE=/tmp/forklift_integrated_smoke.py
fi

if [[ ! -f "$SMOKE_SOURCE" ]]; then
  echo "Forklift integrated smoke script not found: $SMOKE_SOURCE" >&2
  echo "Set FORKLIFT_INTEGRATED_SMOKE_SOURCE or run this script from the repository checkout." >&2
  exit 1
fi

docker cp "$SMOKE_SOURCE" "$ROS_CONTAINER:$SMOKE_SCRIPT"
docker exec -u root "$ROS_CONTAINER" chmod +x "$SMOKE_SCRIPT"

docker exec "$ROS_CONTAINER" bash -lc "
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /jazzy_ws/install/setup.bash
export ROS_DOMAIN_ID='$ROS_DOMAIN_ID'
exec python3 '$SMOKE_SCRIPT'
"
