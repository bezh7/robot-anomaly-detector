from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.derive_features import derive_imu_features
from src.features.feature_contract import DERIVED_IMU_FEATURES

MOTION_SPEED_THRESHOLD = 0.263
VERTICAL_SPEED_THRESHOLD = 0.050
YAW_RATE_THRESHOLD = 0.114

SEVERITY_MULTIPLIERS = {
    "small": 1.5,
    "medium": 3.0,
    "large": 5.0,
}

DURATIONS = {
    "gyro_bias_step": 50,
    "gyro_bias_drift": 75,
    "accel_bias_drift": 75,
    "accel_freeze": 40,
    "gyro_freeze": 40,
    "noise_burst": 20,
    "clipping": 20,
    "hard_clipping": 20,
    "packet_dropout": 20,
    "sensor_lag": 40,
    "timestamp_jitter": 40,
    "cross_axis_leakage": 40,
    "gyro_accel_inconsistency": 40,
    "impact_pulse": 8,
    "vibration_burst": 20,
    "angular_rate_burst": 15,
}

QUAT_COLUMNS = ("q_x", "q_y", "q_z", "q_w")
GYRO_COLUMNS = ("ang_vel_x", "ang_vel_y", "ang_vel_z")
ACCEL_COLUMNS = ("lin_acc_x", "lin_acc_y", "lin_acc_z")


@dataclass(frozen=True)
class InjectedSequence:
    frame: pd.DataFrame
    metadata: dict


def build_injected_sequence(
    frame: pd.DataFrame,
    *,
    context: dict[str, list[float]],
    anomaly_type: str,
    severity: str,
    start_index: int,
    seed: int,
) -> InjectedSequence:
    if anomaly_type in {"impact_pulse", "vibration_burst", "angular_rate_burst"}:
        _require_motion_context(context, start_index=start_index, anomaly_type=anomaly_type)

    if severity not in SEVERITY_MULTIPLIERS:
        raise ValueError(f"unsupported severity: {severity}")
    if anomaly_type not in DURATIONS:
        raise ValueError(f"unsupported anomaly type: {anomaly_type}")

    rng = np.random.default_rng(seed)
    injected = frame.copy(deep=True)
    duration = DURATIONS[anomaly_type]
    end_index = min(len(injected) - 1, start_index + duration - 1)
    multiplier = SEVERITY_MULTIPLIERS[severity]

    if anomaly_type == "gyro_bias_step":
        columns = GYRO_COLUMNS
        target_group = "gyro"
        amplitude = multiplier * _robust_scale(frame, columns)
        for column in columns:
            injected.loc[start_index:end_index, column] += amplitude
    elif anomaly_type == "gyro_bias_drift":
        columns = GYRO_COLUMNS
        target_group = "gyro"
        amplitude = multiplier * _robust_scale(frame, columns)
        ramp = np.linspace(0.0, amplitude, end_index - start_index + 1)
        for column in columns:
            injected.loc[start_index:end_index, column] += ramp
    elif anomaly_type == "accel_bias_drift":
        columns = ACCEL_COLUMNS
        target_group = "accel"
        amplitude = multiplier * _robust_scale(frame, columns)
        ramp = np.linspace(0.0, amplitude, end_index - start_index + 1)
        for column in columns:
            injected.loc[start_index:end_index, column] += ramp
    elif anomaly_type == "accel_freeze":
        columns = ACCEL_COLUMNS
        target_group = "accel"
        for column in columns:
            frozen_value = float(injected.loc[start_index, column])
            injected.loc[start_index:end_index, column] = frozen_value
    elif anomaly_type == "gyro_freeze":
        columns = GYRO_COLUMNS
        target_group = "gyro"
        for column in columns:
            frozen_value = float(injected.loc[start_index, column])
            injected.loc[start_index:end_index, column] = frozen_value
    elif anomaly_type == "noise_burst":
        columns = ACCEL_COLUMNS + GYRO_COLUMNS
        target_group = "mixed"
        amplitude = multiplier * _robust_scale(frame, columns)
        noise = rng.normal(0.0, amplitude, size=(end_index - start_index + 1, len(columns)))
        for column_index, column in enumerate(columns):
            injected.loc[start_index:end_index, column] += noise[:, column_index]
    elif anomaly_type == "clipping":
        columns = ACCEL_COLUMNS
        target_group = "accel"
        clip_value = multiplier * _robust_scale(frame, columns)
        for column in columns:
            injected.loc[start_index:end_index, column] = np.clip(
                injected.loc[start_index:end_index, column],
                -clip_value,
                clip_value,
            )
    elif anomaly_type == "hard_clipping":
        columns = ACCEL_COLUMNS
        target_group = "accel"
        clip_multiplier = {"small": 5.0, "medium": 3.0, "large": 1.5}[severity]
        clip_value = clip_multiplier * _robust_scale(frame, columns)
        for column in columns:
            injected.loc[start_index:end_index, column] = np.clip(
                injected.loc[start_index:end_index, column],
                -clip_value,
                clip_value,
            )
    elif anomaly_type == "packet_dropout":
        columns = QUAT_COLUMNS + GYRO_COLUMNS + ACCEL_COLUMNS
        target_group = "mixed"
        stale_source = injected.loc[max(0, start_index - 1), list(columns)].to_numpy(dtype=float)
        keep_every = {"small": 4, "medium": 2, "large": 1000}[severity]
        for offset, row_index in enumerate(range(start_index, end_index + 1)):
            if severity == "large" or offset % keep_every != 0:
                injected.loc[row_index, list(columns)] = stale_source
                stale_source = injected.loc[max(0, row_index - 1), list(columns)].to_numpy(dtype=float)
    elif anomaly_type == "sensor_lag":
        columns = ACCEL_COLUMNS
        target_group = "accel"
        lag_steps = {"small": 3, "medium": 8, "large": 15}[severity]
        for row_index in range(start_index, end_index + 1):
            source_index = max(0, row_index - lag_steps)
            injected.loc[row_index, list(columns)] = frame.loc[source_index, list(columns)].to_numpy(dtype=float)
    elif anomaly_type == "timestamp_jitter":
        columns = QUAT_COLUMNS + GYRO_COLUMNS + ACCEL_COLUMNS
        target_group = "mixed"
        sigma = {"small": 0.35, "medium": 0.75, "large": 1.5}[severity]
        segment = frame.loc[start_index:end_index, list(columns)].to_numpy(dtype=float)
        positions = np.arange(segment.shape[0], dtype=float)
        increments = np.clip(1.0 + rng.normal(0.0, sigma, size=segment.shape[0]), 0.2, 2.5)
        warped = np.cumsum(increments) - increments[0]
        warped = (warped - warped.min()) / max(1e-6, warped.max() - warped.min()) * max(1.0, segment.shape[0] - 1)
        for column_index, column in enumerate(columns):
            injected.loc[start_index:end_index, column] = np.interp(
                positions,
                warped,
                segment[:, column_index],
            )
    elif anomaly_type == "cross_axis_leakage":
        target_group = "accel"
        alpha = {"small": 0.15, "medium": 0.35, "large": 0.6}[severity]
        segment = frame.loc[start_index:end_index, list(ACCEL_COLUMNS)].to_numpy(dtype=float)
        leaked = segment.copy()
        leaked[:, 0] = segment[:, 0] + alpha * segment[:, 1]
        leaked[:, 1] = segment[:, 1] + alpha * segment[:, 2]
        leaked[:, 2] = segment[:, 2] + alpha * segment[:, 0]
        injected.loc[start_index:end_index, list(ACCEL_COLUMNS)] = leaked
    elif anomaly_type == "gyro_accel_inconsistency":
        target_group = "accel"
        angle_deg = {"small": 15.0, "medium": 35.0, "large": 60.0}[severity]
        theta = np.deg2rad(angle_deg)
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        segment = frame.loc[start_index:end_index, list(ACCEL_COLUMNS)].to_numpy(dtype=float)
        rotated = segment.copy()
        rotated[:, 0] = (segment[:, 0] * cos_t) - (segment[:, 1] * sin_t)
        rotated[:, 1] = (segment[:, 0] * sin_t) + (segment[:, 1] * cos_t)
        injected.loc[start_index:end_index, list(ACCEL_COLUMNS)] = rotated
    elif anomaly_type == "impact_pulse":
        target_group = "accel"
        amplitude = multiplier * _robust_scale(frame, ACCEL_COLUMNS)
        pulse = np.hanning(end_index - start_index + 3)[1:-1] * amplitude
        injected.loc[start_index:end_index, "lin_acc_z"] += pulse
    elif anomaly_type == "vibration_burst":
        target_group = "accel"
        amplitude = multiplier * _robust_scale(frame, ACCEL_COLUMNS)
        phase = np.linspace(0.0, 4.0 * np.pi, end_index - start_index + 1)
        injected.loc[start_index:end_index, "lin_acc_z"] += np.sin(phase) * amplitude
    elif anomaly_type == "angular_rate_burst":
        target_group = "gyro"
        amplitude = multiplier * _robust_scale(frame, GYRO_COLUMNS)
        phase = np.linspace(0.0, 2.0 * np.pi, end_index - start_index + 1)
        injected.loc[start_index:end_index, "ang_vel_z"] += np.sin(phase) * amplitude
    else:
        raise ValueError(f"unsupported anomaly type: {anomaly_type}")

    injected = _refresh_derived_features_if_present(injected)

    return InjectedSequence(
        frame=injected,
        metadata={
            "anomaly_type": anomaly_type,
            "severity": severity,
            "start_index": start_index,
            "end_index": end_index,
            "target_group": target_group,
        },
    )


def _robust_scale(frame: pd.DataFrame, columns: tuple[str, ...]) -> float:
    values = frame.loc[:, list(columns)].to_numpy(dtype=float).reshape(-1)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return mad if mad > 1e-6 else 1.0


def _require_motion_context(context: dict[str, list[float]], *, start_index: int, anomaly_type: str) -> None:
    speed = float(context["gt_speed_mps"][start_index])
    vertical_speed = abs(float(context["gt_vertical_speed_mps"][start_index]))
    yaw_rate = abs(float(context["gt_yaw_rate_rps"][start_index]))

    if anomaly_type in {"impact_pulse", "vibration_burst"}:
        if not (speed > MOTION_SPEED_THRESHOLD or vertical_speed > VERTICAL_SPEED_THRESHOLD):
            raise ValueError("motion anomaly requires moving context")
    elif anomaly_type == "angular_rate_burst":
        if not (speed > MOTION_SPEED_THRESHOLD or yaw_rate > YAW_RATE_THRESHOLD):
            raise ValueError("angular-rate anomaly requires moving or turning context")


def _refresh_derived_features_if_present(frame: pd.DataFrame) -> pd.DataFrame:
    if not all(column in frame.columns for column in DERIVED_IMU_FEATURES):
        return frame
    refreshed = derive_imu_features(frame)
    for column in DERIVED_IMU_FEATURES:
        frame[column] = refreshed[column]
    return frame
