#!/usr/bin/env bash
set -euxo pipefail

: "${IMAGE_URI:?IMAGE_URI is required}"
: "${FEATURE_S3_PREFIX:?FEATURE_S3_PREFIX is required}"
: "${MODELING_S3_PREFIX:?MODELING_S3_PREFIX is required}"

REGISTRY_URI="$(echo "$IMAGE_URI" | cut -d/ -f1)"
AWS_REGION="$(echo "$REGISTRY_URI" | cut -d. -f4)"

mkdir -p /opt/rad/features /opt/rad/modeling
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY_URI"
docker pull "$IMAGE_URI"
aws s3 sync "$FEATURE_S3_PREFIX" /opt/rad/features
docker run --rm --gpus all \
  -v /opt/rad/features:/app/artifacts/features \
  -v /opt/rad/modeling:/app/artifacts/modeling \
  "$IMAGE_URI" \
  -m src.modeling.run_model_search --artifact-root artifacts/features --output-root artifacts/modeling
aws s3 sync /opt/rad/modeling "$MODELING_S3_PREFIX"
