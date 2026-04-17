from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cleaning_contract import GT_CANONICAL_COLUMNS, GT_QUATERNION_COLUMNS, IMU_CANONICAL_COLUMNS, UGV_SEQUENCE_NAMES


def _validate_dataframe_columns(frame: pd.DataFrame, expected_columns: list[str], label: str) -> None:
    if list(frame.columns) != expected_columns:
        raise ValueError(f'{label} columns do not match expected contract')
    if frame.isna().sum().sum() != 0:
        raise ValueError(f'{label} contains NaN values')
    timestamp_deltas = frame['timestamp_ns'].diff().dropna()
    if not timestamp_deltas.gt(0).all():
        raise ValueError(f'{label} timestamps are not strictly increasing')


def _validate_gt_quaternion_continuity(frame: pd.DataFrame, label: str) -> None:
    quaternions = frame[GT_QUATERNION_COLUMNS].to_numpy()
    consecutive_dots = [float(np.dot(previous, current)) for previous, current in zip(quaternions, quaternions[1:])]
    if any(dot < 0.0 for dot in consecutive_dots):
        raise ValueError(f'{label} contains quaternion sign discontinuities')


def validate_clean_dataset(output_dir: Path) -> dict[str, int]:
    raw_manifest = json.loads((output_dir / 'raw_manifest.json').read_text())
    overlap_manifest = json.loads((output_dir / 'overlap_manifest.json').read_text())

    if [row['sequence_name'] for row in raw_manifest] != UGV_SEQUENCE_NAMES:
        raise ValueError('Raw manifest sequence order does not match expected UGV cohort')
    if [row['sequence_name'] for row in overlap_manifest] != UGV_SEQUENCE_NAMES:
        raise ValueError('Overlap manifest sequence order does not match expected UGV cohort')

    for sequence_name in UGV_SEQUENCE_NAMES:
        imu_canonical_path = output_dir / 'imu_canonical' / f'{sequence_name}.parquet'
        gt_canonical_path = output_dir / 'gt_canonical' / f'{sequence_name}.parquet'
        overlap_imu_path = output_dir / 'overlap' / f'{sequence_name}_imu.parquet'
        overlap_gt_path = output_dir / 'overlap' / f'{sequence_name}_gt.parquet'

        imu_canonical = pd.read_parquet(imu_canonical_path)
        gt_canonical = pd.read_parquet(gt_canonical_path)
        overlap_imu = pd.read_parquet(overlap_imu_path)
        overlap_gt = pd.read_parquet(overlap_gt_path)

        _validate_dataframe_columns(imu_canonical, IMU_CANONICAL_COLUMNS, f'{sequence_name} IMU canonical')
        _validate_dataframe_columns(gt_canonical, GT_CANONICAL_COLUMNS, f'{sequence_name} GT canonical')
        _validate_dataframe_columns(overlap_imu, IMU_CANONICAL_COLUMNS, f'{sequence_name} overlap IMU')
        _validate_dataframe_columns(overlap_gt, GT_CANONICAL_COLUMNS, f'{sequence_name} overlap GT')
        _validate_gt_quaternion_continuity(gt_canonical, f'{sequence_name} GT canonical')
        _validate_gt_quaternion_continuity(overlap_gt, f'{sequence_name} overlap GT')

    return {
        'sequence_count': len(UGV_SEQUENCE_NAMES),
        'imu_canonical_count': len(UGV_SEQUENCE_NAMES),
        'gt_canonical_count': len(UGV_SEQUENCE_NAMES),
        'overlap_pair_count': len(UGV_SEQUENCE_NAMES),
    }
