#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?AWS_REGION is required}"
: "${ECR_REPOSITORY_NAME:?ECR_REPOSITORY_NAME is required}"
: "${IMAGE_TAG:?IMAGE_TAG is required}"

ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
REGISTRY_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY_URI}/${ECR_REPOSITORY_NAME}:${IMAGE_TAG}"

if ! aws ecr describe-repositories --repository-names "${ECR_REPOSITORY_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "${ECR_REPOSITORY_NAME}" --region "${AWS_REGION}" >/dev/null
fi

aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${REGISTRY_URI}"
docker buildx build --platform linux/amd64 -f Dockerfile.gpu -t "${IMAGE_URI}" --push .

echo "${IMAGE_URI}"
