# EC2 Manual Run Guide

This guide is the manual operator path for running the modeling pipeline on one persistent NVIDIA EC2 instance.

Use this flow when you want to:

- pull the GPU image once
- run the smoke test
- run Stage 1 search
- run final training
- avoid repeated image pulls and repeated instance launch cycles

## 1. Prerequisites

You need:

- an NVIDIA GPU EC2 instance
- a GPU-ready AMI with Docker and the NVIDIA runtime already working
- an instance profile that can:
  - pull from ECR
  - read feature artifacts from S3
  - write modeling artifacts to S3
- feature artifacts already available in S3
- the published GPU image URI

Recommended instance shape for development:

- `g5.xlarge`

Recommended root volume:

- at least `150 GB`
- preferably `200 GB`

The current published image is large enough that a small root volume is a bad idea.

## 2. Launch One Persistent EC2 Instance

Launch the instance manually from the AWS console or CLI.

Recommended approach:

- keep one instance running during smoke testing and Stage 1 experimentation
- terminate it only after you are done with:
  - smoke test
  - architecture search
  - final training

This avoids repeated image pulls.

## 3. Connect To The Instance

Use SSH or SSM.

Example with SSH:

```bash
ssh -i /path/to/key.pem ubuntu@YOUR_INSTANCE_PUBLIC_DNS
```

Once connected, create local working directories:

```bash
sudo mkdir -p /opt/rad/features /opt/rad/modeling
sudo chown -R "$USER":"$USER" /opt/rad
```

## 4. Pull The GPU Image Once

Set these values first:

```bash
export AWS_REGION=us-east-1
export IMAGE_URI=628161515461.dkr.ecr.us-east-1.amazonaws.com/robot-anomaly-detector:pt26-399abf8-010550
export FEATURE_S3_PREFIX=s3://YOUR_BUCKET/YOUR_PREFIX/artifacts/features
export MODELING_S3_PREFIX=s3://YOUR_BUCKET/YOUR_PREFIX/artifacts/modeling
```

Log in to ECR and pull the image:

```bash
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin 628161515461.dkr.ecr.us-east-1.amazonaws.com

docker pull "$IMAGE_URI"
```

If you reuse the same image tag for multiple runs on the same host, Docker will reuse the local image.

## 5. Sync Feature Artifacts Once

```bash
aws s3 sync "$FEATURE_S3_PREFIX" /opt/rad/features
```

## 6. Run The Smoke Test

This is the smallest real cloud run that proves the end-to-end path:

- image pull works
- features are readable
- training/evaluation runs
- outputs are written

```bash
docker run --rm --gpus all \
  -v /opt/rad/features:/app/artifacts/features \
  -v /opt/rad/modeling:/app/artifacts/modeling \
  "$IMAGE_URI" \
  -m src.modeling.run_model_search \
  --artifact-root artifacts/features \
  --output-root artifacts/modeling \
  --architectures lstm \
  --feature-sets raw \
  --normalizations zscore \
  --batch-sizes 32 \
  --learning-rates 1e-3 \
  --folds fold_2 \
  --trials-per-architecture 1 \
  --max-epochs 1 \
  --seed 7
```

After it completes, sync outputs back to S3:

```bash
aws s3 sync /opt/rad/modeling "$MODELING_S3_PREFIX"
```

## 7. Run Stage 1 Search

This is the main architecture/feature/normalization search.

```bash
docker run --rm --gpus all \
  -v /opt/rad/features:/app/artifacts/features \
  -v /opt/rad/modeling:/app/artifacts/modeling \
  "$IMAGE_URI" \
  -m src.modeling.run_model_search \
  --artifact-root artifacts/features \
  --output-root artifacts/modeling \
  --architectures lstm,tcn \
  --feature-sets raw,raw_plus_derived \
  --normalizations zscore,robust \
  --batch-sizes 32,64,128 \
  --learning-rates 3e-4,1e-3 \
  --folds fold_2,fold_4 \
  --trials-per-architecture 8 \
  --max-epochs 100 \
  --seed 7
```

Sync outputs back to S3 after the run:

```bash
aws s3 sync /opt/rad/modeling "$MODELING_S3_PREFIX"
```

## 8. Inspect Results

Primary artifacts:

- `/opt/rad/modeling/leaderboard.csv`
- `/opt/rad/modeling/leaderboard.json`
- `/opt/rad/modeling/experiment_manifest.json`

Per-experiment outputs include:

- configs
- checkpoints
- train history
- fold metrics
- injected replay metrics
- clean replay metrics

## 9. Run Final Training

Once the final config is chosen, retrain on all dev data:

```bash
docker run --rm --gpus all \
  -v /opt/rad/features:/app/artifacts/features \
  -v /opt/rad/modeling:/app/artifacts/modeling \
  "$IMAGE_URI" \
  -m src.modeling.train_anomaly_model \
  --artifact-root artifacts/features \
  --output-root artifacts/modeling \
  --architecture lstm \
  --feature-set raw \
  --normalization zscore \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --max-epochs 100 \
  --seed 7 \
  --final-train
```

Then sync outputs:

```bash
aws s3 sync /opt/rad/modeling "$MODELING_S3_PREFIX"
```

## 10. Suggested Operator Practices

Use `tmux` or `screen` so your session survives disconnects.

Optional:

```bash
tmux new -s rad
```

Useful checks:

```bash
nvidia-smi
docker image ls
du -sh /opt/rad/features /opt/rad/modeling
```

## 11. Cleanup

When you are done:

- sync any remaining modeling artifacts to S3
- terminate the instance manually

This manual flow is intended for interactive experimentation. The EC2 launcher scripts remain useful later if you want reproducible one-shot job execution.
