from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import subprocess

from src.canonicalize_gt import canonicalize_gt_csv
from src.canonicalize_imu import canonicalize_imu_csv
from src.cleaning_contract import IMU_RAW_FILENAME, UGV_SEQUENCE_NAMES
from src.extract_gt import extract_ground_truth_csv
from src.raw_inventory import build_raw_manifest_from_local_root, build_raw_manifest_from_s3, write_raw_manifest_outputs
from src.trim_overlap import trim_sequence_overlap, write_overlap_manifest_outputs
from src.validate_clean_ugv_dataset import validate_clean_dataset


def _copy_s3_file(s3_path: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['aws', 's3', 'cp', s3_path, str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    return destination


def _process_local_sequence(sequence_name: str, sequence_root: Path, output_dir: Path) -> dict[str, object]:
    gt_zip_candidates = sorted(sequence_root.glob('*_gt.zip'))
    if len(gt_zip_candidates) != 1:
        raise ValueError(f'Expected exactly one GT zip for {sequence_name}')

    gt_raw_dir = output_dir / 'gt_raw' / sequence_name
    gt_csv_path = extract_ground_truth_csv(gt_zip_candidates[0], gt_raw_dir)
    imu_output_path = canonicalize_imu_csv(sequence_name, sequence_root / IMU_RAW_FILENAME, output_dir / 'imu_canonical')
    gt_output_path = canonicalize_gt_csv(sequence_name, gt_csv_path, output_dir / 'gt_canonical')
    return trim_sequence_overlap(sequence_name, imu_output_path, gt_output_path, output_dir / 'overlap')


def build_clean_dataset_from_local_root(
    raw_root: Path,
    output_dir: Path,
    raw_manifest: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = raw_manifest if raw_manifest is not None else build_raw_manifest_from_local_root(raw_root)
    write_raw_manifest_outputs(manifest, output_dir)

    overlap_records = []
    for sequence_name in UGV_SEQUENCE_NAMES:
        overlap_records.append(_process_local_sequence(sequence_name, raw_root / sequence_name, output_dir))

    write_overlap_manifest_outputs(overlap_records, output_dir)
    validate_clean_dataset(output_dir)

    return {
        'sequence_names': UGV_SEQUENCE_NAMES,
        'output_dir': output_dir,
    }


def build_clean_dataset_from_s3(s3_prefix: str, output_dir: Path) -> dict[str, object]:
    normalized_prefix = s3_prefix.rstrip('/') + '/'
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_manifest = build_raw_manifest_from_s3(normalized_prefix)

    with tempfile.TemporaryDirectory() as temporary_directory:
        raw_root = Path(temporary_directory) / 'raw_root'
        raw_root.mkdir(parents=True, exist_ok=True)

        for sequence_name in UGV_SEQUENCE_NAMES:
            sequence_root = raw_root / sequence_name
            sequence_root.mkdir(parents=True, exist_ok=True)
            _copy_s3_file(f'{normalized_prefix}{sequence_name}/{IMU_RAW_FILENAME}', sequence_root / IMU_RAW_FILENAME)
            _copy_s3_file(
                f'{normalized_prefix}{sequence_name}/{sequence_name}_gt.zip',
                sequence_root / f'{sequence_name}_gt.zip',
            )
            calibration_target = sequence_root / 'calibration.yaml'
            try:
                _copy_s3_file(f'{normalized_prefix}{sequence_name}/calibration.yaml', calibration_target)
            except subprocess.CalledProcessError:
                calibration_target.write_text('')

        return build_clean_dataset_from_local_root(
            raw_root=raw_root,
            output_dir=output_dir,
            raw_manifest=raw_manifest,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Build clean UGV IMU and GT artifacts.')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/clean'))
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--s3-prefix', default='s3://<bucket>/<raw-prefix>/')
    args = parser.parse_args()

    if args.raw_root is not None:
        build_clean_dataset_from_local_root(raw_root=args.raw_root, output_dir=args.output_dir)
    else:
        build_clean_dataset_from_s3(s3_prefix=args.s3_prefix, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
