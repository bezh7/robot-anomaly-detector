#!/usr/bin/env bash
set -euo pipefail

docker build -f Dockerfile.cpu -t robot-anomaly-detector-modeling .
mkdir -p artifacts/modeling

docker run --rm \
  -v "$PWD/artifacts/features:/app/artifacts/features" \
  -v "$PWD/artifacts/modeling:/app/artifacts/modeling" \
  robot-anomaly-detector-modeling \
  python -m src.modeling.run_model_search \
    --artifact-root artifacts/features \
    --output-root artifacts/modeling \
    "$@"
