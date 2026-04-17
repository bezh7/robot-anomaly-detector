import numpy as np
import pandas as pd
import pytest

from src.derive_features import align_gt_context_to_feature_grid, derive_imu_features


def make_deterministic_resampled_imu_frame(sample_count: int = 25) -> pd.DataFrame:
    timestamps_ns = (np.arange(sample_count, dtype=np.int64) * 20_000_000).astype(np.int64)
    step = np.arange(sample_count, dtype=float)

    return pd.DataFrame(
        {
            'sequence_name': ['fixture_sequence'] * sample_count,
            'timestamp_ns': timestamps_ns,
            'timestep_index': np.arange(sample_count, dtype=np.int64),
            'q_x': np.zeros(sample_count, dtype=float),
            'q_y': np.zeros(sample_count, dtype=float),
            'q_z': np.zeros(sample_count, dtype=float),
            'q_w': np.ones(sample_count, dtype=float),
            'ang_vel_x': 1.0 + step,
            'ang_vel_y': 2.0 + step,
            'ang_vel_z': 2.0 + step,
            'lin_acc_x': 3.0 + (5.0 * step),
            'lin_acc_y': np.full(sample_count, 4.0, dtype=float),
            'lin_acc_z': np.zeros(sample_count, dtype=float),
        }
    )


def expected_gyro_rms_over_first_25_rows(frame: pd.DataFrame) -> float:
    gyro_norms = np.linalg.norm(frame[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']].to_numpy(), axis=1)
    return float(np.sqrt(np.mean(gyro_norms[:25] ** 2)))


def expected_accel_rms_over_first_25_rows(frame: pd.DataFrame) -> float:
    accel_norms = np.linalg.norm(frame[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].to_numpy(), axis=1)
    return float(np.sqrt(np.mean(accel_norms[:25] ** 2)))


def make_feature_grid_timestamps(duration_s: float, rate_hz: int) -> np.ndarray:
    sample_count = int(duration_s * rate_hz)
    return (np.arange(sample_count, dtype=np.int64) * int(1_000_000_000 / rate_hz)).astype(np.int64)


def make_constant_velocity_constant_yaw_rate_gt_frame(
    gt_rate_hz: int,
    duration_s: float,
    speed_mps: float,
    yaw_rate_rps: float,
) -> pd.DataFrame:
    sample_count = int(duration_s * gt_rate_hz) + 1
    time_s = np.arange(sample_count, dtype=float) / float(gt_rate_hz)
    yaw_rad = yaw_rate_rps * time_s

    return pd.DataFrame(
        {
            'timestamp_ns': (time_s * 1_000_000_000).astype(np.int64),
            'p_w_b_x': speed_mps * time_s,
            'p_w_b_y': np.zeros(sample_count, dtype=float),
            'p_w_b_z': np.zeros(sample_count, dtype=float),
            'q_w_b_x': np.zeros(sample_count, dtype=float),
            'q_w_b_y': np.zeros(sample_count, dtype=float),
            'q_w_b_z': np.sin(yaw_rad / 2.0),
            'q_w_b_w': np.cos(yaw_rad / 2.0),
        }
    )


def test_derive_imu_features_matches_known_norm_derivative_and_rms_values():
    frame = make_deterministic_resampled_imu_frame()

    out = derive_imu_features(frame)

    # The gyro norm must match sqrt(wx^2 + wy^2 + wz^2) exactly for the analytic first row.
    assert out.loc[0, 'gyro_norm'] == pytest.approx(np.sqrt((1.0**2) + (2.0**2) + (2.0**2)))
    # The accel norm must match sqrt(ax^2 + ay^2 + az^2) exactly for the analytic first row.
    assert out.loc[0, 'accel_norm'] == pytest.approx(np.sqrt((3.0**2) + (4.0**2) + (0.0**2)))

    # Backward differences are undefined at row zero, so the implementation must emit an explicit zero boundary value.
    assert out.loc[0, 'angular_accel_norm'] == pytest.approx(0.0)
    # Jerk uses the same backward-difference boundary rule and must also be explicitly zero at row zero.
    assert out.loc[0, 'jerk_norm'] == pytest.approx(0.0)

    # Row one gyro derivative should be ||[1, 1, 1]|| / 0.02 on a 50 Hz grid.
    assert out.loc[1, 'angular_accel_norm'] == pytest.approx(np.sqrt(3.0) / 0.02)
    # Row one accel derivative should be ||[5, 0, 0]|| / 0.02 on a 50 Hz grid.
    assert out.loc[1, 'jerk_norm'] == pytest.approx(5.0 / 0.02)

    # The first fully populated trailing 25-sample gyro RMS must use exactly rows 0..24.
    assert out.loc[24, 'gyro_rms_local'] == pytest.approx(expected_gyro_rms_over_first_25_rows(frame))
    # The first fully populated trailing 25-sample accel RMS must use exactly rows 0..24.
    assert out.loc[24, 'accel_rms_local'] == pytest.approx(expected_accel_rms_over_first_25_rows(frame))

    # All six derived feature columns must be finite and non-null so downstream training does not need ad-hoc cleanup.
    assert (
        out[
            [
                'gyro_norm',
                'accel_norm',
                'angular_accel_norm',
                'jerk_norm',
                'gyro_rms_local',
                'accel_rms_local',
            ]
        ]
        .isna()
        .sum()
        .sum()
        == 0
    )


def test_align_gt_context_recovers_known_speed_and_yaw_rate():
    imu_timestamps_ns = make_feature_grid_timestamps(duration_s=2.0, rate_hz=50)
    gt_frame = make_constant_velocity_constant_yaw_rate_gt_frame(
        gt_rate_hz=5,
        duration_s=2.0,
        speed_mps=2.0,
        yaw_rate_rps=0.5,
    )

    out = align_gt_context_to_feature_grid(imu_timestamps_ns=imu_timestamps_ns, gt_frame=gt_frame)

    # GT context rows must align to the exact IMU feature-grid timestamps with no drift.
    assert np.array_equal(out['timestamp_ns'].to_numpy(), imu_timestamps_ns)
    # Total speed should recover the analytic constant 2.0 m/s trajectory.
    assert out['gt_speed_mps'].dropna().median() == pytest.approx(2.0, rel=1e-6)
    # Horizontal speed should match total speed when motion is confined to x/y.
    assert out['gt_horizontal_speed_mps'].dropna().median() == pytest.approx(2.0, rel=1e-6)
    # Vertical speed should remain exactly zero for a flat trajectory.
    assert out['gt_vertical_speed_mps'].dropna().abs().max() == pytest.approx(0.0, abs=1e-12)
    # Yaw-rate recovery should match the analytic 0.5 rad/s turning profile.
    assert out['gt_yaw_rate_rps'].dropna().median() == pytest.approx(0.5, rel=1e-6)
    # Unwrapped yaw must not jump by ~2pi between neighboring aligned rows.
    assert np.nanmax(np.abs(np.diff(out['gt_yaw_rad'].dropna().to_numpy()))) < np.pi
