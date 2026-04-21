from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.cleaning_contract import IMU_CANONICAL_COLUMNS


RAW_IMU_NUMERIC_COLUMNS = [
    'timestamp',
    'q_x',
    'q_y',
    'q_z',
    'q_w',
    'ang_vel_x',
    'ang_vel_y',
    'ang_vel_z',
    'lin_acc_x',
    'lin_acc_y',
    'lin_acc_z',
]


def canonicalize_imu_csv(sequence_name: str, input_path: Path, output_dir: Path) -> Path:
    frame = pd.read_csv(input_path, skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]

    missing_columns = [column for column in RAW_IMU_NUMERIC_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f'Missing IMU columns: {missing_columns}')

    numeric_frame = frame[RAW_IMU_NUMERIC_COLUMNS].apply(pd.to_numeric, errors='raise')
    numeric_frame = numeric_frame.rename(columns={'timestamp': 'timestamp_ns'})
    numeric_frame['timestamp_ns'] = numeric_frame['timestamp_ns'].astype('int64')

    canonical_frame = numeric_frame.sort_values('timestamp_ns', kind='mergesort').reset_index(drop=True)
    canonical_frame.insert(0, 'sequence_name', sequence_name)
    canonical_frame = canonical_frame[IMU_CANONICAL_COLUMNS]

    if canonical_frame.isna().sum().sum() != 0:
        raise ValueError('NaN values present after IMU canonicalization')

    timestamp_deltas = canonical_frame['timestamp_ns'].diff().dropna()
    if not timestamp_deltas.gt(0).all():
        raise ValueError('IMU timestamps must be strictly increasing after canonicalization')

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{sequence_name}.parquet'
    canonical_frame.to_parquet(output_path, index=False)
    return output_path
