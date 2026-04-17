from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_RATE_HZ = 50.0
DEFAULT_RMS_WINDOW_SAMPLES = 25
NANOSECONDS_PER_SECOND = 1_000_000_000.0

GYRO_COLUMNS = ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']
ACCEL_COLUMNS = ['lin_acc_x', 'lin_acc_y', 'lin_acc_z']
GT_POSITION_COLUMNS = ['p_w_b_x', 'p_w_b_y', 'p_w_b_z']
GT_QUATERNION_COLUMNS = ['q_w_b_x', 'q_w_b_y', 'q_w_b_z', 'q_w_b_w']
GT_CONTEXT_COLUMNS = [
    'gt_speed_mps',
    'gt_horizontal_speed_mps',
    'gt_vertical_speed_mps',
    'gt_yaw_rad',
    'gt_yaw_rate_rps',
]


def _require_columns(frame: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')


def _backward_difference_norm(
    frame: pd.DataFrame,
    value_columns: list[str],
    *,
    timestamp_column: str = 'timestamp_ns',
    default_dt_s: float = 1.0 / DEFAULT_RATE_HZ,
) -> pd.Series:
    component_delta = frame[value_columns].diff()
    if timestamp_column in frame.columns:
        dt_s = frame[timestamp_column].diff().astype(float) / NANOSECONDS_PER_SECOND
    else:
        dt_s = pd.Series(default_dt_s, index=frame.index, dtype=float)

    dt_s = dt_s.where(dt_s > 0.0)
    derivative = component_delta.div(dt_s, axis=0)
    derivative.iloc[0] = 0.0

    norms = np.linalg.norm(derivative.to_numpy(dtype=float), axis=1)
    norms = np.nan_to_num(norms, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(norms, index=frame.index, dtype=float)


def _rolling_rms(values: pd.Series, window_samples: int) -> pd.Series:
    return values.pow(2).rolling(window=window_samples, min_periods=1).mean().pow(0.5)


def derive_imu_features(
    frame: pd.DataFrame,
    *,
    rms_window_samples: int = DEFAULT_RMS_WINDOW_SAMPLES,
) -> pd.DataFrame:
    _require_columns(frame, GYRO_COLUMNS + ACCEL_COLUMNS)

    output = frame.copy()
    output['gyro_norm'] = np.linalg.norm(output[GYRO_COLUMNS].to_numpy(dtype=float), axis=1)
    output['accel_norm'] = np.linalg.norm(output[ACCEL_COLUMNS].to_numpy(dtype=float), axis=1)
    output['angular_accel_norm'] = _backward_difference_norm(output, GYRO_COLUMNS)
    output['jerk_norm'] = _backward_difference_norm(output, ACCEL_COLUMNS)
    output['gyro_rms_local'] = _rolling_rms(output['gyro_norm'], window_samples=rms_window_samples)
    output['accel_rms_local'] = _rolling_rms(output['accel_norm'], window_samples=rms_window_samples)
    return output


def _quaternion_to_unwrapped_yaw(quaternions: np.ndarray) -> np.ndarray:
    x = quaternions[:, 0]
    y = quaternions[:, 1]
    z = quaternions[:, 2]
    w = quaternions[:, 3]

    yaw = np.arctan2(
        2.0 * ((w * z) + (x * y)),
        1.0 - (2.0 * ((y * y) + (z * z))),
    )
    return np.unwrap(yaw)


def _interpolate_with_bounds(
    source_timestamps_ns: np.ndarray,
    source_values: np.ndarray,
    target_timestamps_ns: np.ndarray,
) -> np.ndarray:
    valid_mask = np.isfinite(source_values)
    if valid_mask.sum() == 0:
        return np.full(target_timestamps_ns.shape[0], np.nan, dtype=float)

    valid_timestamps = source_timestamps_ns[valid_mask].astype(float)
    valid_values = source_values[valid_mask].astype(float)

    if valid_timestamps.shape[0] == 1:
        aligned = np.full(target_timestamps_ns.shape[0], np.nan, dtype=float)
        aligned[target_timestamps_ns == int(valid_timestamps[0])] = valid_values[0]
        return aligned

    return np.interp(
        target_timestamps_ns.astype(float),
        valid_timestamps,
        valid_values,
        left=np.nan,
        right=np.nan,
    )


def _derive_gt_context_at_native_rate(gt_frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(gt_frame, ['timestamp_ns'] + GT_POSITION_COLUMNS + GT_QUATERNION_COLUMNS)

    ordered = gt_frame.sort_values('timestamp_ns', kind='mergesort').reset_index(drop=True)
    timestamps_ns = ordered['timestamp_ns'].to_numpy(dtype=np.int64)
    timestamps_s = timestamps_ns.astype(float) / NANOSECONDS_PER_SECOND
    dt_s = pd.Series(timestamps_s).diff().to_numpy(dtype=float)

    position = ordered[GT_POSITION_COLUMNS].to_numpy(dtype=float)
    position_delta = np.diff(position, axis=0, prepend=np.full((1, 3), np.nan))
    velocity = position_delta / dt_s[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    horizontal_speed = np.linalg.norm(velocity[:, :2], axis=1)
    vertical_speed = np.abs(velocity[:, 2])

    yaw_rad = _quaternion_to_unwrapped_yaw(ordered[GT_QUATERNION_COLUMNS].to_numpy(dtype=float))
    yaw_delta = np.diff(yaw_rad, prepend=np.nan)
    yaw_rate_rps = yaw_delta / dt_s

    return pd.DataFrame(
        {
            'timestamp_ns': timestamps_ns,
            'gt_speed_mps': speed,
            'gt_horizontal_speed_mps': horizontal_speed,
            'gt_vertical_speed_mps': vertical_speed,
            'gt_yaw_rad': yaw_rad,
            'gt_yaw_rate_rps': yaw_rate_rps,
        }
    )


def align_gt_context_to_feature_grid(
    *,
    imu_timestamps_ns: np.ndarray,
    gt_frame: pd.DataFrame,
) -> pd.DataFrame:
    imu_timestamps_ns = np.asarray(imu_timestamps_ns, dtype=np.int64)
    native_context = _derive_gt_context_at_native_rate(gt_frame)

    source_timestamps_ns = native_context['timestamp_ns'].to_numpy(dtype=np.int64)
    aligned = pd.DataFrame({'timestamp_ns': imu_timestamps_ns})
    for column in GT_CONTEXT_COLUMNS:
        aligned[column] = _interpolate_with_bounds(
            source_timestamps_ns,
            native_context[column].to_numpy(dtype=float),
            imu_timestamps_ns,
        )
    return aligned
