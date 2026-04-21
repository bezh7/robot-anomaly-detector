from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cleaning_contract import GT_CANONICAL_COLUMNS, GT_QUATERNION_COLUMNS


RAW_GT_NUMERIC_COLUMNS = [
    'timestamp',
    'p_w_b_x',
    'p_w_b_y',
    'p_w_b_z',
    'q_w_b_x',
    'q_w_b_y',
    'q_w_b_z',
    'q_w_b_w',
]


def repair_quaternion_sign_continuity(quaternions: np.ndarray) -> np.ndarray:
    repaired = quaternions.astype(float, copy=True)
    for index in range(1, len(repaired)):
        if float(np.dot(repaired[index - 1], repaired[index])) < 0.0:
            repaired[index] *= -1.0
    return repaired


def canonicalize_gt_csv(sequence_name: str, input_path: Path, output_dir: Path) -> Path:
    frame = pd.read_csv(input_path)
    frame.columns = [str(column).strip() for column in frame.columns]

    missing_columns = [column for column in RAW_GT_NUMERIC_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f'Missing GT columns: {missing_columns}')

    numeric_frame = frame[RAW_GT_NUMERIC_COLUMNS].apply(pd.to_numeric, errors='raise')
    numeric_frame = numeric_frame.rename(columns={'timestamp': 'timestamp_ns'})
    numeric_frame['timestamp_ns'] = numeric_frame['timestamp_ns'].astype('int64')

    canonical_frame = numeric_frame.sort_values('timestamp_ns', kind='mergesort').reset_index(drop=True)
    canonical_frame.loc[:, GT_QUATERNION_COLUMNS] = repair_quaternion_sign_continuity(
        canonical_frame[GT_QUATERNION_COLUMNS].to_numpy()
    )
    canonical_frame.insert(0, 'sequence_name', sequence_name)
    canonical_frame = canonical_frame[GT_CANONICAL_COLUMNS]

    if canonical_frame.isna().sum().sum() != 0:
        raise ValueError('NaN values present after GT canonicalization')

    timestamp_deltas = canonical_frame['timestamp_ns'].diff().dropna()
    if not timestamp_deltas.gt(0).all():
        raise ValueError('GT timestamps must be strictly increasing after canonicalization')

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{sequence_name}.parquet'
    canonical_frame.to_parquet(output_path, index=False)
    return output_path
