from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.features.feature_contract import FEATURE_TABLE_COLUMNS, GT_CONTEXT_FEATURES, RAW_IMU_FEATURES, DERIVED_IMU_FEATURES
from src.features.build_feature_tables import build_feature_table


def _write_clean_imu_parquet(path, *, sequence_name: str) -> None:
    dt_s = 0.005
    timestamps_ns = np.arange(0, int(2.0 / dt_s) + 1, dtype=np.int64) * int(dt_s * 1e9)
    time_s = timestamps_ns.astype(float) / 1e9

    frame = pd.DataFrame(
        {
            'sequence_name': sequence_name,
            'timestamp_ns': timestamps_ns,
            'q_x': np.zeros_like(time_s),
            'q_y': np.zeros_like(time_s),
            'q_z': np.zeros_like(time_s),
            'q_w': np.ones_like(time_s),
            'ang_vel_x': np.sin(2.0 * math.pi * 1.5 * time_s),
            'ang_vel_y': np.cos(2.0 * math.pi * 1.5 * time_s),
            'ang_vel_z': np.sin(2.0 * math.pi * 0.5 * time_s),
            'lin_acc_x': 9.81 + 0.2 * np.sin(2.0 * math.pi * 3.0 * time_s),
            'lin_acc_y': 0.1 * np.cos(2.0 * math.pi * 2.0 * time_s),
            'lin_acc_z': 0.3 * np.sin(2.0 * math.pi * 1.0 * time_s),
        }
    )
    frame.to_parquet(path, index=False)


def _write_clean_gt_parquet(path, *, sequence_name: str) -> None:
    timestamps_ns = np.arange(0, 11, dtype=np.int64) * 200_000_000
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


def test_build_feature_table_writes_contract_quality_not_just_schema(tmp_path):
    imu_path = tmp_path / 'clean_imu.parquet'
    gt_path = tmp_path / 'clean_gt.parquet'
    _write_clean_imu_parquet(imu_path, sequence_name='final_challenge_ugv1')
    _write_clean_gt_parquet(gt_path, sequence_name='final_challenge_ugv1')

    table_path, manifest_record = build_feature_table(
        sequence_name='final_challenge_ugv1',
        imu_input_path=imu_path,
        gt_input_path=gt_path,
        output_dir=tmp_path / 'feature_tables',
    )
    frame = pd.read_parquet(table_path)

    # Column order must be stable so tensor column positions stay deterministic across experiments.
    assert list(frame.columns) == FEATURE_TABLE_COLUMNS
    # Each feature table should represent exactly one source sequence or fold bookkeeping becomes ambiguous.
    assert frame['sequence_name'].nunique() == 1
    assert frame['sequence_name'].iloc[0] == 'final_challenge_ugv1'
    # Strict 20 ms spacing is the core resampling contract that window counts depend on.
    assert np.all(np.diff(frame['timestamp_ns'].to_numpy(dtype=np.int64)) == 20_000_000)
    # Quaternion validity must survive the full table assembly pipeline, not just the raw resampling step.
    assert np.allclose(
        np.linalg.norm(frame[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy(dtype=float), axis=1),
        1.0,
        atol=1e-4,
    )
    # Raw and derived IMU features are direct model/diagnostic inputs, so non-finite values here would poison training.
    assert np.isfinite(frame[RAW_IMU_FEATURES + DERIVED_IMU_FEATURES].to_numpy(dtype=float)).all()
    # GT context may have a warm-up edge, but once support exists it should remain finite for downstream analysis overlays.
    assert frame.loc[10:, GT_CONTEXT_FEATURES].notna().all().all()
    # The manifest must describe the artifact that was actually written, or downstream orchestration will drift from reality.
    assert manifest_record['sequence_name'] == 'final_challenge_ugv1'
    assert manifest_record['row_count'] == len(frame)
    assert manifest_record['target_rate_hz'] == 50
    assert manifest_record['timestamp_start_ns'] == int(frame['timestamp_ns'].iloc[0])
    assert manifest_record['timestamp_end_ns'] == int(frame['timestamp_ns'].iloc[-1])

