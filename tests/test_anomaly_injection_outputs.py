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


def test_accel_bias_drift_targets_accel_group():
    frame = pd.DataFrame(
        {
            "timestamp_ns": list(range(200)),
            "q_x": [0.0] * 200,
            "q_y": [0.0] * 200,
            "q_z": [0.0] * 200,
            "q_w": [1.0] * 200,
            "lin_acc_x": [0.1] * 200,
            "lin_acc_y": [0.2] * 200,
            "lin_acc_z": [0.3] * 200,
            "ang_vel_x": [0.0] * 200,
            "ang_vel_y": [0.0] * 200,
            "ang_vel_z": [0.0] * 200,
        }
    )
    context = {
        "gt_speed_mps": [0.5] * 200,
        "gt_vertical_speed_mps": [0.0] * 200,
        "gt_yaw_rate_rps": [0.0] * 200,
    }

    result = build_injected_sequence(
        frame,
        context=context,
        anomaly_type="accel_bias_drift",
        severity="medium",
        start_index=50,
        seed=11,
    )

    assert result.metadata["target_group"] == "accel"
    assert result.frame.loc[124, "lin_acc_x"] > frame.loc[124, "lin_acc_x"]


def test_gyro_freeze_holds_gyro_values_constant():
    frame = pd.DataFrame(
        {
            "timestamp_ns": list(range(200)),
            "q_x": [0.0] * 200,
            "q_y": [0.0] * 200,
            "q_z": [0.0] * 200,
            "q_w": [1.0] * 200,
            "lin_acc_x": [0.0] * 200,
            "lin_acc_y": [0.0] * 200,
            "lin_acc_z": [0.0] * 200,
            "ang_vel_x": list(range(200)),
            "ang_vel_y": list(range(200)),
            "ang_vel_z": list(range(200)),
        }
    )
    context = {
        "gt_speed_mps": [0.5] * 200,
        "gt_vertical_speed_mps": [0.0] * 200,
        "gt_yaw_rate_rps": [0.2] * 200,
    }

    result = build_injected_sequence(
        frame,
        context=context,
        anomaly_type="gyro_freeze",
        severity="medium",
        start_index=60,
        seed=7,
    )

    assert result.metadata["target_group"] == "gyro"
    assert result.frame.loc[60:99, "ang_vel_x"].nunique() == 1


def test_packet_dropout_stales_raw_samples():
    frame = pd.DataFrame(
        {
            "timestamp_ns": list(range(200)),
            "q_x": [0.0] * 200,
            "q_y": [0.0] * 200,
            "q_z": [0.0] * 200,
            "q_w": [1.0] * 200,
            "lin_acc_x": list(range(200)),
            "lin_acc_y": list(range(200)),
            "lin_acc_z": list(range(200)),
            "ang_vel_x": list(range(200)),
            "ang_vel_y": list(range(200)),
            "ang_vel_z": list(range(200)),
        }
    )
    context = {
        "gt_speed_mps": [0.5] * 200,
        "gt_vertical_speed_mps": [0.0] * 200,
        "gt_yaw_rate_rps": [0.2] * 200,
    }

    result = build_injected_sequence(
        frame,
        context=context,
        anomaly_type="packet_dropout",
        severity="large",
        start_index=80,
        seed=3,
    )

    assert result.metadata["target_group"] == "mixed"
    assert result.frame.loc[80:99, "lin_acc_x"].nunique() == 1


def test_sensor_lag_uses_earlier_accel_values():
    frame = pd.DataFrame(
        {
            "timestamp_ns": list(range(200)),
            "q_x": [0.0] * 200,
            "q_y": [0.0] * 200,
            "q_z": [0.0] * 200,
            "q_w": [1.0] * 200,
            "lin_acc_x": list(range(200)),
            "lin_acc_y": list(range(200)),
            "lin_acc_z": list(range(200)),
            "ang_vel_x": [0.0] * 200,
            "ang_vel_y": [0.0] * 200,
            "ang_vel_z": [0.0] * 200,
        }
    )
    context = {
        "gt_speed_mps": [0.5] * 200,
        "gt_vertical_speed_mps": [0.0] * 200,
        "gt_yaw_rate_rps": [0.2] * 200,
    }

    result = build_injected_sequence(
        frame,
        context=context,
        anomaly_type="sensor_lag",
        severity="medium",
        start_index=100,
        seed=5,
    )

    assert result.metadata["target_group"] == "accel"
    assert result.frame.loc[100, "lin_acc_x"] == frame.loc[92, "lin_acc_x"]


def test_timestamp_jitter_and_cross_axis_leakage_modify_signal():
    frame = pd.DataFrame(
        {
            "timestamp_ns": list(range(200)),
            "q_x": [0.0] * 200,
            "q_y": [0.0] * 200,
            "q_z": [0.0] * 200,
            "q_w": [1.0] * 200,
            "lin_acc_x": [float(i) for i in range(200)],
            "lin_acc_y": [float(i * 2) for i in range(200)],
            "lin_acc_z": [float(i * 3) for i in range(200)],
            "ang_vel_x": [float(i) for i in range(200)],
            "ang_vel_y": [float(i * 2) for i in range(200)],
            "ang_vel_z": [float(i * 3) for i in range(200)],
        }
    )
    context = {
        "gt_speed_mps": [0.5] * 200,
        "gt_vertical_speed_mps": [0.0] * 200,
        "gt_yaw_rate_rps": [0.2] * 200,
    }

    jittered = build_injected_sequence(
        frame,
        context=context,
        anomaly_type="timestamp_jitter",
        severity="medium",
        start_index=70,
        seed=17,
    )
    leaked = build_injected_sequence(
        frame,
        context=context,
        anomaly_type="cross_axis_leakage",
        severity="medium",
        start_index=70,
        seed=17,
    )

    assert jittered.metadata["target_group"] == "mixed"
    assert leaked.metadata["target_group"] == "accel"
    assert not jittered.frame.loc[70:109, "ang_vel_x"].equals(frame.loc[70:109, "ang_vel_x"])
    assert not leaked.frame.loc[70:109, "lin_acc_x"].equals(frame.loc[70:109, "lin_acc_x"])


def test_runtime_inconsistency_rotates_accel_without_touching_gyro():
    frame = pd.DataFrame(
        {
            "timestamp_ns": list(range(200)),
            "q_x": [0.0] * 200,
            "q_y": [0.0] * 200,
            "q_z": [0.0] * 200,
            "q_w": [1.0] * 200,
            "lin_acc_x": [1.0] * 200,
            "lin_acc_y": [0.0] * 200,
            "lin_acc_z": [0.0] * 200,
            "ang_vel_x": [0.3] * 200,
            "ang_vel_y": [0.2] * 200,
            "ang_vel_z": [0.1] * 200,
        }
    )
    context = {
        "gt_speed_mps": [0.5] * 200,
        "gt_vertical_speed_mps": [0.0] * 200,
        "gt_yaw_rate_rps": [0.2] * 200,
    }

    result = build_injected_sequence(
        frame,
        context=context,
        anomaly_type="gyro_accel_inconsistency",
        severity="medium",
        start_index=90,
        seed=9,
    )

    assert result.metadata["target_group"] == "accel"
    assert result.frame.loc[90, "lin_acc_y"] != frame.loc[90, "lin_acc_y"]
    assert result.frame.loc[90, "ang_vel_x"] == frame.loc[90, "ang_vel_x"]
