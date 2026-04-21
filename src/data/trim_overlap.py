from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


def trim_sequence_overlap(sequence_name: str, imu_path: Path, gt_path: Path, output_dir: Path) -> dict[str, object]:
    imu_frame = pd.read_parquet(imu_path)
    gt_frame = pd.read_parquet(gt_path)

    overlap_start_ns = max(int(imu_frame['timestamp_ns'].min()), int(gt_frame['timestamp_ns'].min()))
    overlap_end_ns = min(int(imu_frame['timestamp_ns'].max()), int(gt_frame['timestamp_ns'].max()))
    if overlap_start_ns > overlap_end_ns:
        raise ValueError(f'No overlap found for {sequence_name}')

    trimmed_imu = imu_frame[
        imu_frame['timestamp_ns'].between(overlap_start_ns, overlap_end_ns, inclusive='both')
    ].reset_index(drop=True)
    trimmed_gt = gt_frame[
        gt_frame['timestamp_ns'].between(overlap_start_ns, overlap_end_ns, inclusive='both')
    ].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    imu_output_path = output_dir / f'{sequence_name}_imu.parquet'
    gt_output_path = output_dir / f'{sequence_name}_gt.parquet'
    trimmed_imu.to_parquet(imu_output_path, index=False)
    trimmed_gt.to_parquet(gt_output_path, index=False)

    return {
        'sequence_name': sequence_name,
        'overlap_start_ns': overlap_start_ns,
        'overlap_end_ns': overlap_end_ns,
        'imu_row_count_before': int(len(imu_frame)),
        'imu_row_count_after': int(len(trimmed_imu)),
        'gt_row_count_before': int(len(gt_frame)),
        'gt_row_count_after': int(len(trimmed_gt)),
        'imu_output_path': imu_output_path,
        'gt_output_path': gt_output_path,
    }


def write_overlap_manifest_outputs(records: list[dict[str, object]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'overlap_manifest.json'
    csv_path = output_dir / 'overlap_manifest.csv'

    serializable_records = [
        {
            **record,
            'imu_output_path': str(record['imu_output_path']),
            'gt_output_path': str(record['gt_output_path']),
        }
        for record in records
    ]
    json_path.write_text(json.dumps(serializable_records, indent=2))

    with csv_path.open('w', newline='') as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                'sequence_name',
                'overlap_start_ns',
                'overlap_end_ns',
                'imu_row_count_before',
                'imu_row_count_after',
                'gt_row_count_before',
                'gt_row_count_after',
                'imu_output_path',
                'gt_output_path',
            ],
        )
        writer.writeheader()
        for record in serializable_records:
            writer.writerow(record)

    return {'json': json_path, 'csv': csv_path}
