from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.build_split_manifests import build_dev_split_manifest
from src.build_window_index import build_window_index
from src.feature_contract import (
    DEFAULT_WINDOW_SIZE,
    FEATURE_SET_COLUMNS,
    FEATURE_TABLE_COLUMNS,
    GT_CONTEXT_FEATURES,
    NORMALIZATION_MODES,
    RAW_IMU_FEATURES,
    DERIVED_IMU_FEATURES,
)
from src.fit_normalizers import fit_normalizer
from src.validate_feature_dataset import validate_feature_dataset


DEV_SEQUENCES = ['seq_a', 'seq_b', 'seq_c', 'seq_d']


def make_feature_table(*, sequence_name: str, row_count: int = 220) -> pd.DataFrame:
    timestamps_ns = np.arange(row_count, dtype=np.int64) * 20_000_000
    data: dict[str, object] = {
        'sequence_name': [sequence_name] * row_count,
        'timestamp_ns': timestamps_ns,
        'timestep_index': np.arange(row_count, dtype=np.int64),
        'q_x': np.zeros(row_count, dtype=float),
        'q_y': np.zeros(row_count, dtype=float),
        'q_z': np.zeros(row_count, dtype=float),
        'q_w': np.ones(row_count, dtype=float),
        'ang_vel_x': np.linspace(0.0, 1.0, row_count),
        'ang_vel_y': np.linspace(1.0, 2.0, row_count),
        'ang_vel_z': np.linspace(2.0, 3.0, row_count),
        'lin_acc_x': np.linspace(9.5, 9.8, row_count),
        'lin_acc_y': np.linspace(0.0, 0.5, row_count),
        'lin_acc_z': np.linspace(-0.2, 0.2, row_count),
        'gyro_norm': np.linspace(0.5, 1.5, row_count),
        'accel_norm': np.linspace(9.5, 9.9, row_count),
        'angular_accel_norm': np.linspace(0.0, 0.4, row_count),
        'jerk_norm': np.linspace(0.0, 0.3, row_count),
        'gyro_rms_local': np.linspace(0.5, 1.5, row_count),
        'accel_rms_local': np.linspace(9.5, 9.9, row_count),
        'gt_speed_mps': np.full(row_count, 2.0, dtype=float),
        'gt_horizontal_speed_mps': np.full(row_count, 2.0, dtype=float),
        'gt_vertical_speed_mps': np.zeros(row_count, dtype=float),
        'gt_yaw_rad': np.linspace(0.0, 1.0, row_count),
        'gt_yaw_rate_rps': np.full(row_count, 0.5, dtype=float),
    }
    return pd.DataFrame(data, columns=FEATURE_TABLE_COLUMNS)


def build_sample_feature_artifact_tree(tmp_path: Path) -> Path:
    output_root = tmp_path / 'features'
    feature_tables_dir = output_root / 'feature_tables'
    normalizers_dir = output_root / 'normalizers'
    window_indices_dir = output_root / 'window_indices'
    feature_tables_dir.mkdir(parents=True, exist_ok=True)
    normalizers_dir.mkdir(parents=True, exist_ok=True)
    window_indices_dir.mkdir(parents=True, exist_ok=True)

    feature_tables_by_sequence: dict[str, pd.DataFrame] = {}
    for sequence_name in DEV_SEQUENCES:
        feature_table = make_feature_table(sequence_name=sequence_name)
        feature_table.to_parquet(feature_tables_dir / f'{sequence_name}.parquet', index=False)
        feature_tables_by_sequence[sequence_name] = feature_table

    split_manifest = build_dev_split_manifest(
        dev_sequences=DEV_SEQUENCES,
        feature_sets=list(FEATURE_SET_COLUMNS.keys()),
        normalization_modes=NORMALIZATION_MODES,
        window_size=DEFAULT_WINDOW_SIZE,
    )
    (output_root / 'split_manifest.json').write_text(json.dumps(split_manifest, indent=2, sort_keys=True))

    for fold_index, fold in enumerate(split_manifest['folds'], start=1):
        fold_name = f'fold_{fold_index}'
        training_table = pd.concat(
            [feature_tables_by_sequence[name] for name in fold['training_sequences']],
            axis=0,
            ignore_index=True,
        )
        validation_table = feature_tables_by_sequence[fold['validation_sequence']]

        for experiment in split_manifest['experiments']:
            feature_set_name = experiment['feature_set']
            normalization_mode = experiment['normalization_mode']
            window_size = experiment['window_size']

            fit_normalizer(
                feature_tables=feature_tables_by_sequence,
                training_sequences=fold['training_sequences'],
                feature_columns=FEATURE_SET_COLUMNS[feature_set_name],
                mode=normalization_mode,
                output_path=normalizers_dir / f'{fold_name}_{feature_set_name}_{normalization_mode}.json',
            )

            train_index = build_window_index(
                feature_table=training_table,
                fold_name=fold_name,
                split_name='train',
                feature_set_name=feature_set_name,
                normalization_mode=normalization_mode,
                window_size=window_size,
            )
            train_index.to_parquet(
                window_indices_dir / f'{fold_name}_train_{feature_set_name}_{normalization_mode}.parquet',
                index=False,
            )

            validation_index = build_window_index(
                feature_table=validation_table,
                fold_name=fold_name,
                split_name='validation',
                feature_set_name=feature_set_name,
                normalization_mode=normalization_mode,
                window_size=window_size,
            )
            validation_index.to_parquet(
                window_indices_dir / f'{fold_name}_validation_{feature_set_name}_{normalization_mode}.parquet',
                index=False,
            )

    return output_root


def corrupt_feature_table_spacing(output_root: Path, *, sequence_name: str) -> None:
    table_path = output_root / 'feature_tables' / f'{sequence_name}.parquet'
    feature_table = pd.read_parquet(table_path)
    feature_table.loc[10, 'timestamp_ns'] = int(feature_table.loc[10, 'timestamp_ns']) + 1_000_000
    feature_table.to_parquet(table_path, index=False)


def corrupt_feature_table_quaternion_norm(output_root: Path, *, sequence_name: str) -> None:
    table_path = output_root / 'feature_tables' / f'{sequence_name}.parquet'
    feature_table = pd.read_parquet(table_path)
    feature_table.loc[15, 'q_w'] = 0.2
    feature_table.to_parquet(table_path, index=False)


def test_validate_feature_dataset_accepts_complete_valid_artifact_tree(tmp_path):
    output_root = build_sample_feature_artifact_tree(tmp_path)

    summary = validate_feature_dataset(output_root)

    # A passing validator should summarize the built tree so later failures can be localized quickly.
    assert summary == {
        'sequence_count': 4,
        'feature_table_count': 4,
        'fold_count': 4,
        'experiment_count': 4,
        'normalizer_count': 16,
        'window_index_count': 32,
    }


def test_validate_feature_dataset_rejects_bad_timestamp_spacing_and_bad_quaternion_norm(tmp_path):
    output_root = build_sample_feature_artifact_tree(tmp_path)
    corrupt_feature_table_spacing(output_root, sequence_name='seq_a')
    # Timestamp spacing corruption must fail fast because every downstream window count assumes exact 20 ms steps.
    with pytest.raises(ValueError, match='timestamp spacing'):
        validate_feature_dataset(output_root)

    output_root = build_sample_feature_artifact_tree(tmp_path)
    corrupt_feature_table_quaternion_norm(output_root, sequence_name='seq_b')
    # Quaternion norm corruption must be caught before models learn physically invalid orientation trajectories.
    with pytest.raises(ValueError, match='quaternion'):
        validate_feature_dataset(output_root)
