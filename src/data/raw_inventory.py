from __future__ import annotations

import csv
import json
from pathlib import Path

from src.data.cleaning_contract import CALIBRATION_FILENAME, IMU_RAW_FILENAME, UGV_SEQUENCE_NAMES
from src.common.io_utils import Runner, default_runner


def list_s3_files(s3_prefix: str, runner: Runner = default_runner) -> list[str]:
    output = runner(['aws', 's3', 'ls', s3_prefix])
    files: list[str] = []
    for line in output.splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('PRE '):
            continue
        files.append(stripped_line.split()[-1])
    return files


def build_raw_manifest(listing: dict[str, list[str]]) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []

    for sequence_name in UGV_SEQUENCE_NAMES:
        files = set(listing.get(sequence_name, []))
        manifest.append({
            'sequence_name': sequence_name,
            'has_imu_csv': IMU_RAW_FILENAME in files,
            'has_gt_zip': any(file_name.endswith('_gt.zip') for file_name in files),
            'has_folder_zip': any(file_name.endswith('_folder.zip') for file_name in files),
            'has_calibration': CALIBRATION_FILENAME in files,
            'file_names': sorted(files),
        })

    return manifest


def build_raw_manifest_from_local_root(raw_root: Path) -> list[dict[str, object]]:
    listing = {
        sequence_name: sorted(path.name for path in (raw_root / sequence_name).iterdir())
        for sequence_name in UGV_SEQUENCE_NAMES
    }
    return build_raw_manifest(listing)


def build_raw_manifest_from_s3(s3_prefix: str, runner: Runner = default_runner) -> list[dict[str, object]]:
    normalized_prefix = s3_prefix.rstrip('/') + '/'
    listing = {
        sequence_name: list_s3_files(f'{normalized_prefix}{sequence_name}/', runner=runner)
        for sequence_name in UGV_SEQUENCE_NAMES
    }
    return build_raw_manifest(listing)


def write_raw_manifest_outputs(manifest: list[dict[str, object]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'raw_manifest.json'
    csv_path = output_dir / 'raw_manifest.csv'

    json_path.write_text(json.dumps(manifest, indent=2))

    with csv_path.open('w', newline='') as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=['sequence_name', 'has_imu_csv', 'has_gt_zip', 'has_folder_zip', 'has_calibration', 'file_names'],
        )
        writer.writeheader()
        for row in manifest:
            writer.writerow({
                **row,
                'file_names': json.dumps(row['file_names']),
            })

    return {'json': json_path, 'csv': csv_path}
