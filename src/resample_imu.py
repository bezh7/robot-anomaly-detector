from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from src.feature_contract import DEFAULT_TARGET_RATE_HZ


QUATERNION_COLUMNS = ['q_x', 'q_y', 'q_z', 'q_w']
GYRO_ACCEL_COLUMNS = [
    'ang_vel_x',
    'ang_vel_y',
    'ang_vel_z',
    'lin_acc_x',
    'lin_acc_y',
    'lin_acc_z',
]
OUTPUT_COLUMNS = [
    'sequence_name',
    'timestamp_ns',
    'timestep_index',
    *QUATERNION_COLUMNS,
    *GYRO_ACCEL_COLUMNS,
]


def _estimate_source_rate_hz(timestamps_ns: np.ndarray) -> float:
    deltas_ns = np.diff(timestamps_ns)
    median_delta_ns = float(np.median(deltas_ns))
    if median_delta_ns <= 0:
        raise ValueError('IMU timestamps must be strictly increasing')
    return 1e9 / median_delta_ns


def _build_target_timestamps(*, start_ns: int, end_ns: int, target_rate_hz: int) -> np.ndarray:
    target_step_ns = int(round(1e9 / target_rate_hz))
    if target_step_ns <= 0:
        raise ValueError('target_rate_hz must be positive')
    sample_count = ((end_ns - start_ns) // target_step_ns) + 1
    return start_ns + np.arange(sample_count, dtype=np.int64) * target_step_ns


def _low_pass_gyro_accel(*, values: np.ndarray, source_rate_hz: float, cutoff_hz: float) -> np.ndarray:
    nyquist_hz = 0.5 * source_rate_hz
    effective_cutoff_hz = min(cutoff_hz, 0.99 * nyquist_hz)
    if effective_cutoff_hz <= 0:
        raise ValueError('Invalid low-pass cutoff frequency')
    sos = butter(4, Wn=effective_cutoff_hz, btype='lowpass', fs=source_rate_hz, output='sos')
    return sosfiltfilt(sos, values, axis=0)


def _repair_quaternion_sign_flips(quaternions: np.ndarray) -> np.ndarray:
    repaired = quaternions.copy()
    for index in range(1, repaired.shape[0]):
        if float(np.dot(repaired[index - 1], repaired[index])) < 0.0:
            repaired[index] *= -1.0
    return repaired


def _renormalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError('Encountered zero-norm quaternion during resampling')
    return quaternions / norms


def _resample_single_sequence(
    *,
    sequence_frame: pd.DataFrame,
    target_rate_hz: int,
    low_pass_hz: float,
) -> pd.DataFrame:
    ordered = sequence_frame.sort_values('timestamp_ns', kind='mergesort').reset_index(drop=True)
    timestamps_ns = ordered['timestamp_ns'].to_numpy(dtype=np.int64)

    if timestamps_ns.size < 2:
        raise ValueError('Need at least two IMU rows to resample')
    if not np.all(np.diff(timestamps_ns) > 0):
        raise ValueError('IMU timestamps must be strictly increasing')

    source_rate_hz = _estimate_source_rate_hz(timestamps_ns)
    filtered_gyro_accel = _low_pass_gyro_accel(
        values=ordered[GYRO_ACCEL_COLUMNS].to_numpy(dtype=np.float64),
        source_rate_hz=source_rate_hz,
        cutoff_hz=low_pass_hz,
    )

    repaired_quaternions = _repair_quaternion_sign_flips(
        ordered[QUATERNION_COLUMNS].to_numpy(dtype=np.float64)
    )

    target_timestamps_ns = _build_target_timestamps(
        start_ns=int(timestamps_ns[0]),
        end_ns=int(timestamps_ns[-1]),
        target_rate_hz=target_rate_hz,
    )

    source_t_s = (timestamps_ns - timestamps_ns[0]).astype(np.float64) / 1e9
    target_t_s = (target_timestamps_ns - timestamps_ns[0]).astype(np.float64) / 1e9

    out: dict[str, np.ndarray] = {
        'sequence_name': np.repeat(ordered['sequence_name'].iloc[0], target_timestamps_ns.size),
        'timestamp_ns': target_timestamps_ns,
        'timestep_index': np.arange(target_timestamps_ns.size, dtype=np.int64),
    }

    interpolated_quaternions = np.column_stack(
        [
            np.interp(target_t_s, source_t_s, repaired_quaternions[:, column_idx])
            for column_idx in range(repaired_quaternions.shape[1])
        ]
    )
    interpolated_quaternions = _renormalize_quaternions(interpolated_quaternions)
    for column_idx, column in enumerate(QUATERNION_COLUMNS):
        out[column] = interpolated_quaternions[:, column_idx]

    for column_idx, column in enumerate(GYRO_ACCEL_COLUMNS):
        out[column] = np.interp(target_t_s, source_t_s, filtered_gyro_accel[:, column_idx])

    return pd.DataFrame(out, columns=OUTPUT_COLUMNS)


def resample_imu_frame(
    frame: pd.DataFrame,
    *,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
    low_pass_hz: float = 20.0,
) -> pd.DataFrame:
    required_columns = {'sequence_name', 'timestamp_ns', *QUATERNION_COLUMNS, *GYRO_ACCEL_COLUMNS}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f'Missing IMU columns for resampling: {missing_columns}')

    outputs = []
    for _, sequence_frame in frame.groupby('sequence_name', sort=False):
        outputs.append(
            _resample_single_sequence(
                sequence_frame=sequence_frame,
                target_rate_hz=target_rate_hz,
                low_pass_hz=low_pass_hz,
            )
        )
    return pd.concat(outputs, ignore_index=True)
