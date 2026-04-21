from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from src.features.feature_contract import FEATURE_SET_COLUMNS


NormalizerLookupKey = tuple[str, str, str]


def normalizer_lookup_key(fold_name: str, feature_set_name: str, normalization_mode: str) -> NormalizerLookupKey:
    return (fold_name, feature_set_name, normalization_mode)


class WindowDataset:
    def __init__(
        self,
        *,
        feature_tables: Mapping[str, pd.DataFrame],
        window_index: pd.DataFrame,
        normalizer_params: Mapping[NormalizerLookupKey, Mapping[str, object]],
    ) -> None:
        self.feature_tables = dict(feature_tables)
        self.window_index = window_index.reset_index(drop=True)
        self.normalizer_params = dict(normalizer_params)

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, object]]:
        window_record = self.window_index.iloc[int(index)]
        sequence_name = str(window_record['sequence_name'])
        feature_set_name = str(window_record['feature_set_name'])
        fold_name = str(window_record['fold_name'])
        normalization_mode = str(window_record['normalization_mode'])
        start_row = int(window_record['start_row'])
        end_row = int(window_record['end_row'])

        if feature_set_name not in FEATURE_SET_COLUMNS:
            raise KeyError(f'Unknown feature set: {feature_set_name}')
        if sequence_name not in self.feature_tables:
            raise KeyError(f'Missing feature table for sequence {sequence_name!r}')

        key = normalizer_lookup_key(fold_name, feature_set_name, normalization_mode)
        if key not in self.normalizer_params:
            raise KeyError(f'Missing normalizer params for key {key!r}')
        params = self.normalizer_params[key]

        feature_columns = FEATURE_SET_COLUMNS[feature_set_name]
        if list(params['feature_columns']) != list(feature_columns):
            raise ValueError('Normalizer feature columns do not match requested feature set')

        frame = self.feature_tables[sequence_name].reset_index(drop=True)
        window = frame.iloc[start_row : end_row + 1][feature_columns].to_numpy(dtype=float)
        center = np.asarray([float(params['center'][column]) for column in feature_columns], dtype=float)
        scale = np.asarray([float(params['scale'][column]) for column in feature_columns], dtype=float)
        normalized = (window - center) / scale

        return normalized, window_record.to_dict()
