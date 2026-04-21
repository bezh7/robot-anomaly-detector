import pandas as pd
import pytest

from src.modeling.anomaly_injection import build_injected_sequence


def test_impact_pulse_targets_accel_channels_and_preserves_length():
    frame = pd.DataFrame(
        {
            "timestamp_ns": list(range(200)),
            "lin_acc_x": [0.0] * 200,
            "lin_acc_y": [0.0] * 200,
            "lin_acc_z": [0.0] * 200,
            "ang_vel_x": [0.0] * 200,
            "ang_vel_y": [0.0] * 200,
            "ang_vel_z": [0.0] * 200,
        }
    )
    context = {
        "gt_speed_mps": [0.7] * 200,
        "gt_vertical_speed_mps": [0.1] * 200,
        "gt_yaw_rate_rps": [0.0] * 200,
    }

    result = build_injected_sequence(
        frame,
        context=context,
        anomaly_type="impact_pulse",
        severity="medium",
        start_index=100,
        seed=13,
    )

    assert len(result.frame) == 200
    assert result.metadata["target_group"] == "accel"
    assert result.frame["lin_acc_z"].abs().max() > 0.0


def test_gyro_bias_step_targets_gyro_group():
    frame = pd.DataFrame(
        {
            "timestamp_ns": list(range(200)),
            "lin_acc_x": [0.0] * 200,
            "lin_acc_y": [0.0] * 200,
            "lin_acc_z": [0.0] * 200,
            "ang_vel_x": [0.0] * 200,
            "ang_vel_y": [0.0] * 200,
            "ang_vel_z": [0.0] * 200,
        }
    )
    context = {
        "gt_speed_mps": [0.7] * 200,
        "gt_vertical_speed_mps": [0.0] * 200,
        "gt_yaw_rate_rps": [0.2] * 200,
    }

    result = build_injected_sequence(
        frame,
        context=context,
        anomaly_type="gyro_bias_step",
        severity="small",
        start_index=50,
        seed=7,
    )

    assert result.metadata["target_group"] == "gyro"
    assert result.frame["ang_vel_x"].abs().max() > 0.0


def test_motion_anomaly_requires_motion_context():
    frame = pd.DataFrame(
        {
            "timestamp_ns": list(range(200)),
            "lin_acc_x": [0.0] * 200,
            "lin_acc_y": [0.0] * 200,
            "lin_acc_z": [0.0] * 200,
            "ang_vel_x": [0.0] * 200,
            "ang_vel_y": [0.0] * 200,
            "ang_vel_z": [0.0] * 200,
        }
    )
    context = {
        "gt_speed_mps": [0.0] * 200,
        "gt_vertical_speed_mps": [0.0] * 200,
        "gt_yaw_rate_rps": [0.0] * 200,
    }

    with pytest.raises(ValueError):
        build_injected_sequence(
            frame,
            context=context,
            anomaly_type="impact_pulse",
            severity="medium",
            start_index=100,
            seed=13,
        )


def test_injection_refreshes_derived_features_when_present():
    frame = pd.DataFrame(
        {
            "timestamp_ns": list(range(200)),
            "lin_acc_x": [0.0] * 200,
            "lin_acc_y": [0.0] * 200,
            "lin_acc_z": [0.0] * 200,
            "ang_vel_x": [0.0] * 200,
            "ang_vel_y": [0.0] * 200,
            "ang_vel_z": [0.0] * 200,
            "gyro_norm": [0.0] * 200,
            "accel_norm": [0.0] * 200,
            "angular_accel_norm": [0.0] * 200,
            "jerk_norm": [0.0] * 200,
            "gyro_rms_local": [0.0] * 200,
            "accel_rms_local": [0.0] * 200,
        }
    )
    context = {
        "gt_speed_mps": [0.7] * 200,
        "gt_vertical_speed_mps": [0.1] * 200,
        "gt_yaw_rate_rps": [0.0] * 200,
    }

    result = build_injected_sequence(
        frame,
        context=context,
        anomaly_type="impact_pulse",
        severity="medium",
        start_index=100,
        seed=13,
    )

    assert result.frame["accel_norm"].abs().max() > 0.0
