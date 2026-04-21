#!/usr/bin/env bash
set -euxo pipefail

: "${IMAGE_URI:?IMAGE_URI is required}"
: "${FEATURE_S3_PREFIX:?FEATURE_S3_PREFIX is required}"
: "${MODELING_S3_PREFIX:?MODELING_S3_PREFIX is required}"
: "${ARCHITECTURE:=lstm}"
: "${FEATURE_SET:=raw}"
: "${NORMALIZATION:=zscore}"
: "${BATCH_SIZE:=64}"
: "${LEARNING_RATE:=1e-3}"
: "${MAX_EPOCHS:=100}"
: "${SEED:=7}"

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
  -m src.modeling.train_anomaly_model \
  --artifact-root artifacts/features \
  --output-root artifacts/modeling \
  --architecture "$ARCHITECTURE" \
  --feature-set "$FEATURE_SET" \
  --normalization "$NORMALIZATION" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --max-epochs "$MAX_EPOCHS" \
  --seed "$SEED" \
  --final-train
aws s3 sync /opt/rad/modeling "$MODELING_S3_PREFIX"
