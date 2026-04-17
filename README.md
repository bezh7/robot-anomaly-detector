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
from src.io_utils import default_runner
from src.profiling import build_sequence_manifest, write_manifest_outputs

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

from src.cohort_analysis import analyze_rc_vs_ugv, build_motion_signature_manifest
from src.io_utils import default_runner

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
