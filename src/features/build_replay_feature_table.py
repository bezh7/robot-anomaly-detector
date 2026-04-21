from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.features.build_feature_tables import build_feature_table


def build_replay_feature_table(
    *,
    clean_root: Path | str,
    feature_root: Path | str,
    sequence_name: str,
) -> Path:
    clean_root = Path(clean_root)
    feature_root = Path(feature_root)

    overlap_dir = clean_root / "overlap"
    imu_input_path = overlap_dir / f"{sequence_name}_imu.parquet"
    gt_input_path = overlap_dir / f"{sequence_name}_gt.parquet"
    if not imu_input_path.exists():
        raise FileNotFoundError(f"missing IMU overlap parquet: {imu_input_path}")
    if not gt_input_path.exists():
        raise FileNotFoundError(f"missing GT overlap parquet: {gt_input_path}")

    feature_tables_dir = feature_root / "feature_tables"
    feature_tables_dir.mkdir(parents=True, exist_ok=True)
    output_path, manifest_record = build_feature_table(
        sequence_name=sequence_name,
        imu_input_path=imu_input_path,
        gt_input_path=gt_input_path,
        output_dir=feature_tables_dir,
    )

    manifest_path = feature_root / "feature_table_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = []

    manifest = [entry for entry in manifest if entry["sequence_name"] != sequence_name]
    manifest.append(manifest_record)
    manifest.sort(key=lambda entry: str(entry["sequence_name"]))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a replay feature table for a held-out/demo sequence.")
    parser.add_argument("--clean-root", default="artifacts/demo_clean", help="Root containing overlap clean artifacts.")
    parser.add_argument("--feature-root", default="artifacts/replay_features_real", help="Feature artifact root to update.")
    parser.add_argument("--sequence-name", required=True, help="Sequence name to build.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = build_replay_feature_table(
        clean_root=args.clean_root,
        feature_root=args.feature_root,
        sequence_name=args.sequence_name,
    )
    print(output_path)


if __name__ == "__main__":
    main()
