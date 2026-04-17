from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_window_index import WINDOW_INDEX_COLUMNS
from src.feature_contract import (
    DERIVED_IMU_FEATURES,
    FEATURE_SET_COLUMNS,
    FEATURE_TABLE_COLUMNS,
    NORMALIZATION_MODES,
    RAW_IMU_FEATURES,
)

EXPECTED_TIMESTAMP_STEP_NS = 20_000_000
QUATERNION_COLUMNS = ['q_x', 'q_y', 'q_z', 'q_w']


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _validate_feature_table(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if list(frame.columns) != FEATURE_TABLE_COLUMNS:
        raise ValueError(f'feature table columns do not match contract for {path.name}')
    if frame['sequence_name'].nunique() != 1:
        raise ValueError(f'feature table {path.name} mixes multiple sequences')
    deltas = np.diff(frame['timestamp_ns'].to_numpy(dtype=np.int64))
    if not np.all(deltas > 0):
        raise ValueError(f'feature table {path.name} timestamps are not strictly increasing')
    if not np.all(deltas == EXPECTED_TIMESTAMP_STEP_NS):
        raise ValueError(f'feature table {path.name} violates timestamp spacing contract')
    quaternion_norms = np.linalg.norm(frame[QUATERNION_COLUMNS].to_numpy(dtype=float), axis=1)
    if not np.allclose(quaternion_norms, 1.0, atol=1e-4):
        raise ValueError(f'feature table {path.name} violates quaternion norm contract')
    if not np.isfinite(frame[RAW_IMU_FEATURES + DERIVED_IMU_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(f'feature table {path.name} contains non-finite model or diagnostic values')
    return frame


def _expected_fold_names(folds: list[dict[str, object]]) -> list[str]:
    return [f'fold_{index}' for index in range(1, len(folds) + 1)]


def _normalizer_filename(fold_name: str, feature_set_name: str, normalization_mode: str) -> str:
    return f'{fold_name}_{feature_set_name}_{normalization_mode}.json'


def _window_index_filename(fold_name: str, split_name: str, feature_set_name: str, normalization_mode: str) -> str:
    return f'{fold_name}_{split_name}_{feature_set_name}_{normalization_mode}.parquet'


def _validate_normalizer(
    payload: dict[str, object],
    *,
    training_sequences: list[str],
    feature_set_name: str,
    normalization_mode: str,
) -> None:
    if payload['mode'] != normalization_mode:
        raise ValueError('normalizer mode does not match requested experiment')
    if payload['fitted_sequence_names'] != training_sequences:
        raise ValueError('normalizer fitted sequence provenance does not match split manifest')
    expected_columns = FEATURE_SET_COLUMNS[feature_set_name]
    if payload['feature_columns'] != expected_columns:
        raise ValueError('normalizer feature columns do not match feature set contract')
    if sorted(payload['center'].keys()) != sorted(expected_columns):
        raise ValueError('normalizer center keys do not cover the expected feature columns')
    if sorted(payload['scale'].keys()) != sorted(expected_columns):
        raise ValueError('normalizer scale keys do not cover the expected feature columns')


def _validate_window_index(
    index_frame: pd.DataFrame,
    *,
    split_name: str,
    fold_name: str,
    feature_set_name: str,
    normalization_mode: str,
    allowed_sequences: list[str],
    feature_tables: dict[str, pd.DataFrame],
) -> None:
    if list(index_frame.columns) != WINDOW_INDEX_COLUMNS:
        raise ValueError('window index columns do not match contract')
    if not index_frame['split_name'].eq(split_name).all():
        raise ValueError('window index split_name column is inconsistent')
    if not index_frame['fold_name'].eq(fold_name).all():
        raise ValueError('window index fold_name column is inconsistent')
    if not index_frame['feature_set_name'].eq(feature_set_name).all():
        raise ValueError('window index feature_set_name column is inconsistent')
    if not index_frame['normalization_mode'].eq(normalization_mode).all():
        raise ValueError('window index normalization_mode column is inconsistent')

    for record in index_frame.to_dict(orient='records'):
        sequence_name = str(record['sequence_name'])
        if sequence_name not in allowed_sequences:
            raise ValueError('window index references a sequence outside the declared split')
        if sequence_name not in feature_tables:
            raise ValueError('window index references a missing feature table')

        feature_table = feature_tables[sequence_name].reset_index(drop=True)
        start_row = int(record['start_row'])
        end_row = int(record['end_row'])
        window_size = int(record['window_size'])
        if start_row < 0 or end_row >= len(feature_table) or end_row < start_row:
            raise ValueError('window index row bounds are invalid')
        if end_row - start_row + 1 != window_size:
            raise ValueError('window index window size does not match row bounds')
        if int(record['start_timestamp_ns']) != int(feature_table.loc[start_row, 'timestamp_ns']):
            raise ValueError('window index start timestamp does not match source table')
        if int(record['end_timestamp_ns']) != int(feature_table.loc[end_row, 'timestamp_ns']):
            raise ValueError('window index end timestamp does not match source table')


def validate_feature_dataset(output_root: Path) -> dict[str, int]:
    output_root = Path(output_root)
    feature_tables_dir = output_root / 'feature_tables'
    normalizers_dir = output_root / 'normalizers'
    window_indices_dir = output_root / 'window_indices'
    split_manifest_path = output_root / 'split_manifest.json'

    split_manifest = _load_json(split_manifest_path)
    folds = split_manifest['folds']
    experiments = split_manifest['experiments']
    fold_names = _expected_fold_names(folds)

    feature_tables: dict[str, pd.DataFrame] = {}
    for path in sorted(feature_tables_dir.glob('*.parquet')):
        frame = _validate_feature_table(path)
        feature_tables[str(frame['sequence_name'].iloc[0])] = frame

    normalizer_paths = sorted(normalizers_dir.glob('*.json'))
    window_index_paths = sorted(window_indices_dir.glob('*.parquet'))

    expected_normalizer_count = len(folds) * len(experiments)
    if len(normalizer_paths) != expected_normalizer_count:
        raise ValueError('normalizer artifact count does not match fold/experiment matrix')
    expected_window_index_count = len(folds) * len(experiments) * 2
    if len(window_index_paths) != expected_window_index_count:
        raise ValueError('window index artifact count does not match fold/experiment matrix')

    for fold_name, fold in zip(fold_names, folds, strict=True):
        training_sequences = list(fold['training_sequences'])
        validation_sequence = str(fold['validation_sequence'])
        for experiment in experiments:
            feature_set_name = str(experiment['feature_set'])
            normalization_mode = str(experiment['normalization_mode'])
            if normalization_mode not in NORMALIZATION_MODES:
                raise ValueError('split manifest references unsupported normalization mode')
            if feature_set_name not in FEATURE_SET_COLUMNS:
                raise ValueError('split manifest references unsupported feature set')

            normalizer_path = normalizers_dir / _normalizer_filename(fold_name, feature_set_name, normalization_mode)
            if not normalizer_path.exists():
                raise ValueError(f'missing normalizer artifact: {normalizer_path.name}')
            _validate_normalizer(
                _load_json(normalizer_path),
                training_sequences=training_sequences,
                feature_set_name=feature_set_name,
                normalization_mode=normalization_mode,
            )

            train_index_path = window_indices_dir / _window_index_filename(
                fold_name, 'train', feature_set_name, normalization_mode
            )
            validation_index_path = window_indices_dir / _window_index_filename(
                fold_name, 'validation', feature_set_name, normalization_mode
            )
            if not train_index_path.exists() or not validation_index_path.exists():
                raise ValueError('missing window index artifact for fold/experiment')

            _validate_window_index(
                pd.read_parquet(train_index_path),
                split_name='train',
                fold_name=fold_name,
                feature_set_name=feature_set_name,
                normalization_mode=normalization_mode,
                allowed_sequences=training_sequences,
                feature_tables=feature_tables,
            )
            _validate_window_index(
                pd.read_parquet(validation_index_path),
                split_name='validation',
                fold_name=fold_name,
                feature_set_name=feature_set_name,
                normalization_mode=normalization_mode,
                allowed_sequences=[validation_sequence],
                feature_tables=feature_tables,
            )

    return {
        'sequence_count': len(feature_tables),
        'feature_table_count': len(feature_tables),
        'fold_count': len(folds),
        'experiment_count': len(experiments),
        'normalizer_count': len(normalizer_paths),
        'window_index_count': len(window_index_paths),
    }
