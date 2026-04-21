# robot-anomaly-detector

## Configuration

Replace the placeholder S3 prefixes below with your own storage locations before
running the pipeline.

- Raw dataset prefix: `s3://<bucket>/<raw-prefix>/`
- Artifact prefix: `s3://<bucket>/<artifact-prefix>/`

## Restore Clean Artifacts For Feature Builds

Use this when running `build_feature_dataset` and `artifacts/clean` is missing.

```bash
aws s3 sync s3://<bucket>/<artifact-prefix>/clean artifacts/clean
```

The raw downloaded sequence prefix should point at your mirrored dataset root:
- `s3://<bucket>/<raw-prefix>/`

At the moment, each downloaded sequence contains only `imu_data.csv`, so the first profiling pass is IMU-only.

### Re-run basic profiling

```bash
python3 - <<'PY'
from pathlib import Path
from src.common.io_utils import default_runner
from src.data.profiling import build_sequence_manifest, write_manifest_outputs

manifest = build_sequence_manifest(
    's3://<bucket>/<raw-prefix>/',
    runner=default_runner,
)
write_manifest_outputs(manifest, Path('artifacts/profiling'))
PY
```

### Re-run RC-vs-UGV motion analysis

```bash
python3 - <<'PY'
import csv
import json
from pathlib import Path

from src.data.cohort_analysis import analyze_rc_vs_ugv, build_motion_signature_manifest
from src.common.io_utils import default_runner

output_dir = Path('artifacts/profiling')
output_dir.mkdir(parents=True, exist_ok=True)

signatures = build_motion_signature_manifest(
    's3://<bucket>/<raw-prefix>/',
    runner=default_runner,
)
report = analyze_rc_vs_ugv(signatures)

with (output_dir / 'motion_signatures.csv').open('w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=list(signatures[0].keys()))
    writer.writeheader()
    writer.writerows(signatures)

(output_dir / 'rc_vs_ugv_analysis.json').write_text(json.dumps(report, indent=2))
PY
```

### Outputs

Profiling writes:
- `artifacts/profiling/sequence_manifest.csv`
- `artifacts/profiling/sequence_manifest.json`
- `artifacts/profiling/motion_signatures.csv`
- `artifacts/profiling/rc_vs_ugv_analysis.json`

These artifacts summarize:
- sequence name
- platform hint inferred from folder name
- row count and duration
- estimated sample rate
- duplicate or non-monotonic timestamps
- per-column missing-value counts
- per-sequence IMU motion signatures
- RC-vs-UGV cohort compatibility

## Local Python Environment

For the data and modeling pipelines, create a local virtual environment and
install the pinned repo dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Local Modeling Search

Run the stage-one search pipeline directly:

```bash
python -m src.modeling.run_model_search \
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

Outputs land under:

- `artifacts/modeling/<experiment_id>/<fold_name>/...`
- `artifacts/modeling/leaderboard.csv`
- `artifacts/modeling/leaderboard.json`
- `artifacts/modeling/experiment_manifest.json`

## Docker Modeling Search

For local Docker runs, use the CPU image:

```bash
scripts/run_model_search_in_docker.sh
```

This requires a running local Docker daemon.

This mounts:

- `artifacts/features` into `/app/artifacts/features`
- `artifacts/modeling` into `/app/artifacts/modeling`

Extra CLI arguments are forwarded to `python -m src.modeling.run_model_search`.

The CPU container uses:

- `Dockerfile.cpu`
- `requirements-runtime.txt`
- a CPU-only PyTorch install from `https://download.pytorch.org/whl/cpu`

For GPU-backed Docker hosts such as EC2, build the GPU image instead:

```bash
docker build -f Dockerfile.gpu -t robot-anomaly-detector-modeling-gpu .
docker run --rm --gpus all \
  -v "$PWD/artifacts/features:/app/artifacts/features" \
  -v "$PWD/artifacts/modeling:/app/artifacts/modeling" \
  robot-anomaly-detector-modeling-gpu \
  -m src.modeling.run_model_search \
  --artifact-root artifacts/features \
  --output-root artifacts/modeling
```

The GPU container uses an official AWS Deep Learning Container base image:

- `public.ecr.aws/deep-learning-containers/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker`

The local CPU Docker path is aligned to the same PyTorch major/minor version
(`2.6.0`). Build `Dockerfile.gpu` on a compatible Linux NVIDIA GPU host. The
local CPU Docker path is the intended default for Apple Silicon and other
CPU-only machines.

The GPU image sets `ENTRYPOINT ["python"]`, so GPU `docker run` commands pass
module arguments directly.

## Publish GPU Image To ECR

For NVIDIA-backed EC2 runs, publish the GPU image first:

```bash
AWS_REGION=us-east-1 \
ECR_REPOSITORY_NAME=robot-anomaly-detector \
IMAGE_TAG=git-$(git rev-parse --short HEAD) \
bash scripts/publish_gpu_image_to_ecr.sh
```

This prints the final image URI. Use that image URI in the EC2 launcher commands below.

## EC2 Experiment Job

Launch a one-shot EC2 search/evaluation job using the prebuilt ECR image:

```bash
python -m src.modeling.launch_ec2_experiment_job \
  --image-uri <account>.dkr.ecr.<region>.amazonaws.com/robot-anomaly-detector:<tag> \
  --feature-s3-prefix s3://<bucket>/<artifact-prefix>/features \
  --modeling-s3-prefix s3://<bucket>/<artifact-prefix>/modeling \
  --ami-id <ami-id> \
  --instance-type g4dn.xlarge \
  --instance-profile-name <instance-profile> \
  --key-name <key-name> \
  --security-group-id <sg-id> \
  --subnet-id <subnet-id>
```

Use this for:

- stage-one random search
- stage-two fold evaluation
- stage-three repeated-seed evaluation

Expected EC2 environment:

- Linux NVIDIA GPU instance such as `g4dn` or `g5`
- GPU-ready AMI with Docker + NVIDIA runtime already working
- instance profile with:
  - ECR pull access
  - S3 read/write access

## EC2 Final Training Job

Launch a separate one-shot EC2 final-training job after model selection:

```bash
python -m src.modeling.launch_ec2_final_training_job \
  --image-uri <account>.dkr.ecr.<region>.amazonaws.com/robot-anomaly-detector:<tag> \
  --feature-s3-prefix s3://<bucket>/<artifact-prefix>/features \
  --modeling-s3-prefix s3://<bucket>/<artifact-prefix>/modeling-final \
  --ami-id <ami-id> \
  --instance-type g4dn.xlarge \
  --instance-profile-name <instance-profile> \
  --key-name <key-name> \
  --security-group-id <sg-id> \
  --subnet-id <subnet-id> \
  --architecture lstm \
  --feature-set raw \
  --normalization zscore \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --max-epochs 100 \
  --seed 7
```
