from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_contract import FEATURE_SET_COLUMNS, RAW_IMU_FEATURES
from src.features.window_dataset import WindowDataset, normalizer_lookup_key


def sample_feature_tables_by_sequence() -> dict[str, pd.DataFrame]:
    row_count = 150
    data: dict[str, object] = {
        'sequence_name': ['seq_a'] * row_count,
        'timestamp_ns': np.arange(row_count, dtype=np.int64) * 20_000_000,
        'timestep_index': np.arange(row_count, dtype=np.int64),
    }
    for column_index, column_name in enumerate(RAW_IMU_FEATURES):
        data[column_name] = np.arange(row_count, dtype=float) + float(column_index)
    return {'seq_a': pd.DataFrame(data)}


def sample_window_index() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'fold_name': 'fold_1',
                'split_name': 'train',
                'feature_set_name': 'raw',
                'normalization_mode': 'zscore',
                'sequence_name': 'seq_a',
                'window_size': 150,
                'stride': 50,
                'start_row': 0,
                'end_row': 149,
                'start_timestamp_ns': 0,
                'end_timestamp_ns': 149 * 20_000_000,
            }
        ]
    )


def sample_normalizer_params() -> dict[tuple[str, str, str], dict[str, object]]:
    return {
        normalizer_lookup_key('fold_1', 'raw', 'zscore'): {
            'mode': 'zscore',
            'feature_columns': FEATURE_SET_COLUMNS['raw'],
            'center': {column_name: float(column_index) for column_index, column_name in enumerate(RAW_IMU_FEATURES)},
            'scale': {column_name: 2.0 for column_name in RAW_IMU_FEATURES},
        }
    }


def expected_normalized_window_tensor() -> np.ndarray:
    per_row_value = np.arange(150, dtype=float)[:, None] / 2.0
    return np.repeat(per_row_value, len(RAW_IMU_FEATURES), axis=1)


def test_window_dataset_returns_expected_normalized_tensor_and_metadata():
    dataset = WindowDataset(
        feature_tables=sample_feature_tables_by_sequence(),
        window_index=sample_window_index(),
        normalizer_params=sample_normalizer_params(),
    )

    x, meta = dataset[0]
    expected = expected_normalized_window_tensor()

    # Tensor shape must match the feature-set dimensionality and fixed window length used during training.
    assert x.shape == expected.shape == (150, 10)
    # Exact normalized values matter because model behavior changes if any channel uses the wrong center or scale.
    assert np.allclose(x, expected)
    # Metadata must trace tensors back to source windows so attributions and debugging map to the correct time slice.
    assert meta['sequence_name'] == 'seq_a'
    assert meta['feature_set_name'] == 'raw'
    assert meta['start_row'] == 0
    assert meta['end_row'] == 149

