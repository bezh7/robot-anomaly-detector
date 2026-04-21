# robot-anomaly-detector

This repository contains an end-to-end anomaly-detection pipeline for UGV IMU
telemetry: feature building, model search, evaluation, EC2/Docker execution,
final training, and a replay dashboard for demo review.

The current state of the project is intentionally candid. The modeling pipeline
is runnable and the experiment loop is documented, but the replay demo also
shows where the current detector works and where it fails. That is part of the
artifact: this repo is meant to show engineering decisions, test evidence,
model-selection rationale, and the next technical steps.

## Current Status

- Feature pipeline, model search, evaluation, Docker packaging, EC2 launchers,
  and replay dashboard are implemented.
- Final selected model:
  - `lstm / raw_plus_derived / zscore / bs64 / lr1e-3`
- Final model training completed on EC2 and artifacts were synced.
- Replay dashboard runs locally against the held-out run
  `final_challenge_ugv2`.
- The replay demo currently exposes a real limitation:
  - the selected model plus current score/thresholding setup does not reliably
    flag the subtle synthetic anomalies we injected into the held-out run.

## What Was Built

- **Data pipeline**
  - sequence profiling
  - clean overlap artifacts
  - feature tables
  - normalization artifacts
  - fold manifests
- **Modeling pipeline**
  - LSTM autoencoder
  - temporally compressed TCN autoencoder
  - model search
  - fold evaluation
  - leaderboard generation
  - final full-dev training
- **Execution layer**
  - local Python runs
  - CPU Docker runs
  - GPU Docker image for EC2
  - ECR publish flow
  - EC2 experiment/final-train launchers
- **Demo layer**
  - replay dashboard
  - held-out replay source build
  - synthetic replay injection for controlled demos

## Model Selection Summary

The project used a three-stage selection process, recorded in
`notes/notes.md`.

- **Stage 1**
  - representative-fold search on `fold_2` and `fold_4`
  - result: the leaderboard saturated at the top, so Stage 1 narrowed the
    search space but did not strongly separate the best configs
- **Stage 2**
  - four-fold finalist evaluation
  - result: `tcn/raw/zscore/bs64/lr1e-3` was the strongest single-seed config
- **Stage 3**
  - repeated-seed stability check on the top two configs
  - result: `lstm/raw_plus_derived/zscore/bs64/lr1e-3` was more stable across
    seeds `7, 17, 27`

Selected final model:

- `lstm / raw_plus_derived / zscore / bs64 / lr1e-3`

Key recorded results from `notes/notes.md`:

- **Stage 2**
  - `tcn-raw-zscore-bs64-lr1e-03-seed7`
    - detection `1.0`
    - median TTD `0.18 s`
    - clean alerts/min `0.0`
    - group attribution top1 `1.0`
  - `lstm-raw_plus_derived-zscore-bs64-lr1e-03-seed7`
    - detection `1.0`
    - median TTD `0.18 s`
    - clean alerts/min `0.0`
    - group attribution top1 `0.9643`
- **Stage 3**
  - `lstm-raw_plus_derived-zscore-bs64-lr1e-03-seeds7-17-27`
    - detection `1.0`
    - median TTD `0.18 s`
    - clean alerts/min `0.0`
    - group attribution top1 `0.9643`
  - `tcn-raw-zscore-bs64-lr1e-03-seeds7-17-27`
    - detection `0.9896`
    - median TTD `0.18 s`
    - clean alerts/min `0.0`
    - group attribution top1 `0.9762`

Decision rationale:

- the TCN won the strongest single-seed four-fold sweep
- the LSTM won on repeated-seed stability
- for a limited-data setting, we chose stability over the slightly stronger
  single-seed attribution result

## What Worked

- The experiment pipeline is reproducible end-to-end.
- The EC2 + ECR + Docker path works.
- The final model trains and restores correctly.
- The held-out replay dashboard runs against real artifacts.
- The final selected LSTM performs well on the synthetic fold benchmark used in
  model selection.

Selected Stage 3 per-anomaly results from `notes/notes.md`:

- `gyro_bias_drift`
  - detection `1.0`
  - attribution `0.8333`
  - median TTD `0.58 s`
- `accel_freeze`
  - detection `1.0`
  - attribution `0.9167`
  - median TTD `0.58 s`
- `clipping`
  - detection `1.0`
  - attribution `1.0`
  - median TTD `0.18 s`

These are synthetic benchmark results across `4 folds × 3 seeds`, not labels
from naturally annotated incidents in the raw dataset.

## What Failed Or Is Still Weak

The replay demo on the held-out run (`final_challenge_ugv2`) exposed a real
problem:

- the current score/thresholding setup is not good enough for subtle held-out
  anomaly replay
- the detector produces clustered clean-run alerts in some accel-dominant
  regions
- the injected demo anomalies are visible on the chart, but the current alert
  logic does not reliably fire on them

This is not just a threshold bug.

Current evidence suggests:

- pooled residual scoring is too blunt
- subtle anomalies like drift/freeze do not move the current scalar score much
- clipping does affect the `accel` group score, but the scalar thresholding path
  buries it

So the most likely next technical step is not just “tune the percentile.” It is
to replace or augment the current score with something stronger:

- group-aware residual scores
- dynamic / state-aware thresholding
- change-detection style scoring
- forecasting-based detectors for subtle temporal faults

## Demo

Recorded candid dashboard demo:

- [`_assets/v1_demo.mov`](_assets/v1_demo.mov)

Attempted inline embed below. If GitHub does not render it inline, use the link
above.

<video src="_assets/v1_demo.mov" controls muted width="100%"></video>

What the demo shows:

- a real held-out replay source
- current dashboard playback
- synthetic anomaly overlays
- places where the model/score path fails to promote subtle anomalies into
  alerts

## Verification

Before packaging this branch, the repo was rechecked against the current branch
state rather than assuming it was still green from earlier work.

The current verification target is:

- local test suite
- replay dashboard import/smoke path

Current branch verification result:

- `69 passed` via `pytest -q`

The branch also includes targeted replay-dashboard tests in:

- `tests/test_replay_dashboard_data_outputs.py`

## Notes And Justifications

The main experiment and decision log lives in:

- `notes/notes.md`

That file records:

- search-space choices
- Stage 1, 2, and 3 outcomes
- final model selection rationale
- replay demo preparation
- subtle-anomaly benchmark metrics
- thresholding and held-out replay failure analysis

## Next Steps

If this project were continued, the next technical work should be:

1. replace the current scalar residual score with a stronger anomaly score
2. test group-aware and change-detection style scoring
3. add at least one forecasting-style detector for drift/freeze
4. revisit the replay dashboard against the improved detector
5. collect more varied real runs before making stronger architecture claims

The current repo is therefore best understood as:

- a working anomaly-detection pipeline
- a documented experiment log
- a candid demo of both what worked and what still needs improvement

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

## Replay Dashboard

If final-model artifacts are missing locally, restore them first:

```bash
aws s3 sync s3://robot-anomaly-detector-628161515461-us-east-1-20260421/artifacts/modeling/final artifacts/modeling_final
```

Run the final-model replay dashboard locally:

```bash
streamlit run dashboard/replay_app.py
```

Expected local artifact roots:

- `artifacts/replay_features_real`
- `artifacts/modeling_final`

To prepare the real held-out replay source (`final_challenge_ugv2`):

```bash
aws s3 sync s3://robot-anomaly-detector-data/workspaces/surabaya-v1/artifacts/features artifacts/replay_features_real
aws s3 sync s3://robot-anomaly-detector-data/workspaces/surabaya-v1/artifacts/clean/overlap artifacts/demo_clean/overlap --exclude "*" --include "final_challenge_ugv2_imu.parquet" --include "final_challenge_ugv2_gt.parquet"
python -m src.features.build_replay_feature_table --clean-root artifacts/demo_clean --feature-root artifacts/replay_features_real --sequence-name final_challenge_ugv2
```

The dashboard is intentionally minimal. It loads the final selected model,
recomputes replay scores in memory, and shows:

- anomaly score vs threshold
- current rolling sensor window
- grouped attribution scores
- a simple 10 Hz play mode for recording replay walkthroughs
