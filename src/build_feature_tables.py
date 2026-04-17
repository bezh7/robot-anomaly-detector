from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.derive_features import GT_CONTEXT_COLUMNS, align_gt_context_to_feature_grid, derive_imu_features
from src.feature_contract import DEFAULT_TARGET_RATE_HZ, FEATURE_TABLE_COLUMNS
from src.resample_imu import resample_imu_frame


def _load_sequence_frame(input_path: Path, *, sequence_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(input_path)
    if 'sequence_name' not in frame.columns:
        raise ValueError(f'Missing sequence_name column in {input_path}')
    unique_sequences = frame['sequence_name'].dropna().unique().tolist()
    if unique_sequences != [sequence_name]:
        raise ValueError(
            f'Expected only sequence {sequence_name!r} in {input_path}, found {unique_sequences!r}'
        )
    return frame


def build_feature_table(
    *,
    sequence_name: str,
    imu_input_path: Path,
    gt_input_path: Path,
    output_dir: Path,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
) -> tuple[Path, dict[str, object]]:
    imu_input_path = Path(imu_input_path)
    gt_input_path = Path(gt_input_path)
    output_dir = Path(output_dir)

    imu_frame = _load_sequence_frame(imu_input_path, sequence_name=sequence_name)
    gt_frame = _load_sequence_frame(gt_input_path, sequence_name=sequence_name)

    resampled_imu = resample_imu_frame(imu_frame, target_rate_hz=target_rate_hz)
    imu_with_features = derive_imu_features(resampled_imu)
    gt_context = align_gt_context_to_feature_grid(
        imu_timestamps_ns=imu_with_features['timestamp_ns'].to_numpy(dtype='int64'),
        gt_frame=gt_frame,
    )

    feature_table = imu_with_features.copy()
    for column in GT_CONTEXT_COLUMNS:
        feature_table[column] = gt_context[column].to_numpy()
    feature_table = feature_table.loc[:, FEATURE_TABLE_COLUMNS]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{sequence_name}.parquet'
    feature_table.to_parquet(output_path, index=False)

    manifest_record: dict[str, object] = {
        'sequence_name': sequence_name,
        'feature_table_path': str(output_path),
        'row_count': int(len(feature_table)),
        'target_rate_hz': int(target_rate_hz),
        'timestamp_start_ns': int(feature_table['timestamp_ns'].iloc[0]),
        'timestamp_end_ns': int(feature_table['timestamp_ns'].iloc[-1]),
        'imu_input_path': str(imu_input_path),
        'gt_input_path': str(gt_input_path),
    }

    return output_path, manifest_record
