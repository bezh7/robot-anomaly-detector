from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.build_pre_model_qa_report import build_pre_model_qa_report, compute_fold_feature_drift, compute_sequence_feature_stats
from src.build_split_manifests import build_dev_split_manifest
from src.build_window_index import build_window_index
from src.feature_contract import FEATURE_SET_COLUMNS, FEATURE_TABLE_COLUMNS
from src.fit_normalizers import fit_normalizer


def _make_feature_table(sequence_name: str, *, offset: float, gt_missing_rows: int = 0) -> pd.DataFrame:
    row_count = 180
    timestamps_ns = np.arange(row_count, dtype=np.int64) * 20_000_000
    base = np.arange(row_count, dtype=float)
    gt_speed = np.full(row_count, 2.0 + offset, dtype=float)
    if gt_missing_rows:
        gt_speed[-gt_missing_rows:] = np.nan

    return pd.DataFrame(
        {
            'sequence_name': [sequence_name] * row_count,
            'timestamp_ns': timestamps_ns,
            'timestep_index': np.arange(row_count, dtype=np.int64),
            'q_x': np.zeros(row_count, dtype=float),
            'q_y': np.zeros(row_count, dtype=float),
            'q_z': np.zeros(row_count, dtype=float),
            'q_w': np.ones(row_count, dtype=float),
            'ang_vel_x': np.sin(base / 25.0) + offset,
            'ang_vel_y': np.cos(base / 25.0),
            'ang_vel_z': np.sin(base / 40.0),
            'lin_acc_x': 9.81 + 0.1 * np.sin(base / 10.0),
            'lin_acc_y': 0.1 * np.cos(base / 11.0),
            'lin_acc_z': 0.2 * np.sin(base / 14.0),
            'gyro_norm': np.linspace(0.5, 1.5, row_count),
            'accel_norm': np.linspace(9.7, 9.9, row_count),
            'angular_accel_norm': np.linspace(0.0, 0.2, row_count),
            'jerk_norm': np.linspace(0.0, 0.3, row_count),
            'gyro_rms_local': np.linspace(0.5, 1.5, row_count),
            'accel_rms_local': np.linspace(9.7, 9.9, row_count),
            'gt_speed_mps': gt_speed,
            'gt_horizontal_speed_mps': np.full(row_count, 2.0 + offset, dtype=float),
            'gt_vertical_speed_mps': np.zeros(row_count, dtype=float),
            'gt_yaw_rad': np.linspace(0.0, 1.0, row_count),
            'gt_yaw_rate_rps': np.full(row_count, 0.5, dtype=float),
        },
        columns=FEATURE_TABLE_COLUMNS,
    )


def _make_raw_overlap_imu(sequence_name: str, *, offset: float) -> pd.DataFrame:
    dt_s = 0.005
    row_count = 721
    timestamps_ns = np.arange(row_count, dtype=np.int64) * int(dt_s * 1e9)
    t = timestamps_ns.astype(float) / 1e9
    return pd.DataFrame(
        {
            'sequence_name': [sequence_name] * row_count,
            'timestamp_ns': timestamps_ns,
            'q_x': np.zeros(row_count, dtype=float),
            'q_y': np.zeros(row_count, dtype=float),
            'q_z': np.zeros(row_count, dtype=float),
            'q_w': np.ones(row_count, dtype=float),
            'ang_vel_x': np.sin(2 * np.pi * 1.0 * t) + offset,
            'ang_vel_y': np.cos(2 * np.pi * 1.0 * t),
            'ang_vel_z': np.sin(2 * np.pi * 0.5 * t),
            'lin_acc_x': 9.81 + 0.1 * np.sin(2 * np.pi * 2.0 * t),
            'lin_acc_y': 0.1 * np.cos(2 * np.pi * 2.0 * t),
            'lin_acc_z': 0.2 * np.sin(2 * np.pi * 1.0 * t),
        }
    )


def _build_feature_root(tmp_path: Path) -> tuple[Path, Path]:
    feature_root = tmp_path / 'features'
    clean_root = tmp_path / 'clean'
    (feature_root / 'feature_tables').mkdir(parents=True, exist_ok=True)
    (feature_root / 'normalizers').mkdir(parents=True, exist_ok=True)
    (feature_root / 'window_indices').mkdir(parents=True, exist_ok=True)
    (clean_root / 'overlap').mkdir(parents=True, exist_ok=True)

    feature_tables = {
        'seq_a': _make_feature_table('seq_a', offset=0.0, gt_missing_rows=0),
        'seq_b': _make_feature_table('seq_b', offset=10.0, gt_missing_rows=10),
    }

    for sequence_name, frame in feature_tables.items():
        frame.to_parquet(feature_root / 'feature_tables' / f'{sequence_name}.parquet', index=False)
        _make_raw_overlap_imu(sequence_name, offset=0.0 if sequence_name == 'seq_a' else 10.0).to_parquet(
            clean_root / 'overlap' / f'{sequence_name}_imu.parquet', index=False
        )

    split_manifest = build_dev_split_manifest(
        dev_sequences=['seq_a', 'seq_b'],
        feature_sets=['raw'],
        normalization_modes=['zscore'],
        window_size=150,
    )
    (feature_root / 'split_manifest.json').write_text(json.dumps(split_manifest, indent=2, sort_keys=True))

    for fold_index, fold in enumerate(split_manifest['folds'], start=1):
        fold_name = f'fold_{fold_index}'
        fit_normalizer(
            feature_tables=feature_tables,
            training_sequences=fold['training_sequences'],
            feature_columns=FEATURE_SET_COLUMNS['raw'],
            mode='zscore',
            output_path=feature_root / 'normalizers' / f'{fold_name}_raw_zscore.json',
        )

        train_table = pd.concat([feature_tables[name] for name in fold['training_sequences']], ignore_index=True)
        val_table = feature_tables[fold['validation_sequence']]
        build_window_index(
            feature_table=train_table,
            fold_name=fold_name,
            split_name='train',
            feature_set_name='raw',
            normalization_mode='zscore',
            window_size=150,
        ).to_parquet(feature_root / 'window_indices' / f'{fold_name}_train_raw_zscore.parquet', index=False)
        build_window_index(
            feature_table=val_table,
            fold_name=fold_name,
            split_name='validation',
            feature_set_name='raw',
            normalization_mode='zscore',
            window_size=150,
        ).to_parquet(feature_root / 'window_indices' / f'{fold_name}_validation_raw_zscore.parquet', index=False)

    return feature_root, clean_root


def test_compute_sequence_feature_stats_tracks_missingness_and_duration():
    feature_tables = {
        'seq_a': _make_feature_table('seq_a', offset=0.0, gt_missing_rows=0),
        'seq_b': _make_feature_table('seq_b', offset=10.0, gt_missing_rows=10),
    }

    stats = compute_sequence_feature_stats(feature_tables)

    ang_vel_seq_a = stats[(stats['sequence_name'] == 'seq_a') & (stats['feature_name'] == 'ang_vel_x')].iloc[0]
    gt_seq_b = stats[(stats['sequence_name'] == 'seq_b') & (stats['feature_name'] == 'gt_speed_mps')].iloc[0]

    # Duration should come from exact timestamp span so later window/latency reasoning uses true sequence length.
    assert ang_vel_seq_a['duration_s'] == pytest.approx((179 * 20_000_000) / 1e9)
    # Feature summaries must preserve exact descriptive math for downstream QA comparisons.
    expected_ang_vel = np.sin(np.arange(180, dtype=float) / 25.0)
    assert ang_vel_seq_a['mean'] == pytest.approx(np.mean(expected_ang_vel))
    assert ang_vel_seq_a['p50'] == pytest.approx(np.median(expected_ang_vel))
    # Missingness is a first-class quality signal; GT gaps must be quantified instead of silently ignored.
    assert gt_seq_b['missing_rate'] == pytest.approx(10 / 180)



def test_build_pre_model_qa_report_writes_summary_tables_and_plots(tmp_path):
    feature_root, clean_root = _build_feature_root(tmp_path)

    output_root, summary = build_pre_model_qa_report(
        feature_root=feature_root,
        output_root=tmp_path / 'qa',
        clean_root=clean_root,
    )

    drift = pd.read_csv(output_root / 'fold_feature_drift.csv')
    window_counts = pd.read_csv(output_root / 'fold_window_counts.csv')
    qa_summary = (output_root / 'qa_summary.md').read_text()

    train_vals = np.sin(np.arange(180, dtype=float) / 25.0) + 10.0
    val_vals = np.sin(np.arange(180, dtype=float) / 25.0)
    expected_smd = abs(np.mean(train_vals) - np.mean(val_vals)) / np.std(train_vals, ddof=0)
    drift_row = drift[(drift['fold_name'] == 'fold_1') & (drift['feature_name'] == 'ang_vel_x')].iloc[0]

    # Summary should reflect the actual artifact tree that was inspected, not hard-coded expectations.
    assert summary['sequence_count'] == 2
    assert summary['fold_count'] == 2
    assert summary['plot_count'] == 6
    # Drift math should remain interpretable and reproducible for pre-model train/validation slice review.
    assert drift_row['standardized_mean_diff'] == pytest.approx(expected_smd)
    # Window count outputs should preserve split bookkeeping for every generated index artifact.
    assert set(window_counts['split_name']) == {'train', 'validation'}
    # Markdown report must contain a recommendation so the QA pass ends with a concrete go/no-go signal.
    assert 'Recommendation:' in qa_summary
    # The spot-check plot bundle should exist for both sequences and all three plot families.
    assert len(list((output_root / 'plots').glob('*.png'))) == 6
