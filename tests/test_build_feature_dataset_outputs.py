from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.build_feature_dataset import build_feature_dataset
from src.feature_contract import DERIVED_IMU_FEATURES, RAW_IMU_FEATURES


RESTORE_CLEAN_COMMAND = (
    'aws s3 sync '
    's3://<bucket>/<artifact-prefix>/clean '
    'artifacts/clean'
)


def _write_clean_imu_parquet(path: Path, *, sequence_name: str, phase_shift: float) -> None:
    dt_s = 0.005
    timestamps_ns = np.arange(0, int(4.0 / dt_s) + 1, dtype=np.int64) * int(dt_s * 1e9)
    time_s = timestamps_ns.astype(float) / 1e9

    frame = pd.DataFrame(
        {
            'sequence_name': sequence_name,
            'timestamp_ns': timestamps_ns,
            'q_x': np.zeros_like(time_s),
            'q_y': np.zeros_like(time_s),
            'q_z': np.zeros_like(time_s),
            'q_w': np.ones_like(time_s),
            'ang_vel_x': np.sin(2.0 * math.pi * (1.0 + phase_shift) * time_s),
            'ang_vel_y': np.cos(2.0 * math.pi * (1.5 + phase_shift) * time_s),
            'ang_vel_z': np.sin(2.0 * math.pi * (0.5 + phase_shift) * time_s),
            'lin_acc_x': 9.81 + 0.2 * np.sin(2.0 * math.pi * (3.0 + phase_shift) * time_s),
            'lin_acc_y': 0.1 * np.cos(2.0 * math.pi * (2.0 + phase_shift) * time_s),
            'lin_acc_z': 0.3 * np.sin(2.0 * math.pi * (1.0 + phase_shift) * time_s),
        }
    )
    frame.to_parquet(path, index=False)


def _write_clean_gt_parquet(path: Path, *, sequence_name: str) -> None:
    timestamps_ns = np.arange(0, 21, dtype=np.int64) * 200_000_000
    time_s = timestamps_ns.astype(float) / 1e9
    yaw = 0.5 * time_s

    frame = pd.DataFrame(
        {
            'sequence_name': sequence_name,
            'timestamp_ns': timestamps_ns,
            'p_w_b_x': 2.0 * time_s,
            'p_w_b_y': np.zeros_like(time_s),
            'p_w_b_z': np.zeros_like(time_s),
            'q_w_b_x': np.zeros_like(time_s),
            'q_w_b_y': np.zeros_like(time_s),
            'q_w_b_z': np.sin(yaw / 2.0),
            'q_w_b_w': np.cos(yaw / 2.0),
        }
    )
    frame.to_parquet(path, index=False)


def sample_clean_root(tmp_path: Path) -> Path:
    clean_root = tmp_path / 'clean'
    overlap_dir = clean_root / 'overlap'
    overlap_dir.mkdir(parents=True, exist_ok=True)

    for index, sequence_name in enumerate(['seq_a', 'seq_b', 'seq_c', 'seq_d']):
        _write_clean_imu_parquet(
            overlap_dir / f'{sequence_name}_imu.parquet',
            sequence_name=sequence_name,
            phase_shift=0.1 * index,
        )
        _write_clean_gt_parquet(
            overlap_dir / f'{sequence_name}_gt.parquet',
            sequence_name=sequence_name,
        )

    return clean_root


def test_build_feature_dataset_reports_missing_clean_inputs_with_actionable_message(tmp_path):
    missing_root = tmp_path / 'missing_clean'

    with pytest.raises(FileNotFoundError) as excinfo:
        build_feature_dataset(clean_root=missing_root, output_root=tmp_path / 'features')

    message = str(excinfo.value)

    assert str(missing_root) in message
    assert 'clean artifact layer is required' in message.lower()
    assert RESTORE_CLEAN_COMMAND in message


def test_build_feature_dataset_produces_quality_checked_artifacts_for_fixture_sequences(tmp_path):
    output_root = build_feature_dataset(
        clean_root=sample_clean_root(tmp_path),
        output_root=tmp_path / 'features',
        dev_sequences=['seq_a', 'seq_b', 'seq_c', 'seq_d'],
    )

    feature_table = pd.read_parquet(output_root / 'feature_tables' / 'seq_a.parquet')
    split_manifest = json.loads((output_root / 'split_manifest.json').read_text())
    window_index = pd.read_parquet(output_root / 'window_indices' / 'fold_1_train_raw_zscore.parquet')

    # The built table must satisfy the core feature-layer contract end-to-end, not just intermediate helpers.
    assert np.all(np.diff(feature_table['timestamp_ns'].to_numpy(dtype=np.int64)) == 20_000_000)
    # Model-facing raw features and derived diagnostics must all be finite after the full orchestration pipeline.
    assert np.isfinite(feature_table[RAW_IMU_FEATURES + DERIVED_IMU_FEATURES].to_numpy(dtype=float)).all()
    # The split manifest should encode exactly the agreed four phase-1 experiments.
    assert len(split_manifest['experiments']) == 4
    # Default training windows must reflect the agreed 3 s / 1 s configuration or training volume drifts silently.
    assert int(window_index['window_size'].iloc[0]) == 150
    assert int(window_index['stride'].iloc[0]) == 50
