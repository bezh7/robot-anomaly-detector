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
    "accel_freeze": 40,
    "noise_burst": 20,
    "clipping": 20,
    "impact_pulse": 8,
    "vibration_burst": 20,
    "angular_rate_burst": 15,
}

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
    elif anomaly_type == "accel_freeze":
        columns = ACCEL_COLUMNS
        target_group = "accel"
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
