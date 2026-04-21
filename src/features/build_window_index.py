from __future__ import annotations

import pandas as pd


WINDOW_INDEX_COLUMNS = [
    'fold_name',
    'split_name',
    'feature_set_name',
    'normalization_mode',
    'sequence_name',
    'window_size',
    'stride',
    'start_row',
    'end_row',
    'start_timestamp_ns',
    'end_timestamp_ns',
]


def build_window_index(
    *,
    feature_table: pd.DataFrame,
    fold_name: str,
    split_name: str,
    feature_set_name: str,
    normalization_mode: str,
    window_size: int,
    stride: int | None = None,
    train_stride: int = 50,
    inference_stride: int = 5,
) -> pd.DataFrame:
    if window_size <= 0:
        raise ValueError('window_size must be positive')

    selected_stride = stride
    if selected_stride is None:
        selected_stride = train_stride if split_name == 'train' else inference_stride
    if selected_stride <= 0:
        raise ValueError('stride must be positive')

    records: list[dict[str, int | str]] = []

    for sequence_name, sequence_frame in feature_table.groupby('sequence_name', sort=False):
        sequence_frame = sequence_frame.reset_index(drop=True)
        row_count = len(sequence_frame)
        if row_count < window_size:
            continue

        window_count = ((row_count - window_size) // selected_stride) + 1

        for window_idx in range(window_count):
            start_row = window_idx * selected_stride
            end_row = start_row + window_size - 1
            records.append(
                {
                    'fold_name': fold_name,
                    'split_name': split_name,
                    'feature_set_name': feature_set_name,
                    'normalization_mode': normalization_mode,
                    'sequence_name': str(sequence_name),
                    'window_size': window_size,
                    'stride': selected_stride,
                    'start_row': start_row,
                    'end_row': end_row,
                    'start_timestamp_ns': int(sequence_frame.loc[start_row, 'timestamp_ns']),
                    'end_timestamp_ns': int(sequence_frame.loc[end_row, 'timestamp_ns']),
                }
            )

    return pd.DataFrame.from_records(records, columns=WINDOW_INDEX_COLUMNS)
