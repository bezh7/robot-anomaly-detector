from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.resample_imu import resample_imu_frame


def make_synthetic_imu_frame(*, source_rate_hz: int, duration_s: float, accel_x) -> pd.DataFrame:
    sample_count = int(source_rate_hz * duration_s)
    dt_ns = int(round(1e9 / source_rate_hz))
    timestamps_ns = np.arange(sample_count, dtype=np.int64) * dt_ns
    t = timestamps_ns / 1e9

    return pd.DataFrame(
        {
            'sequence_name': ['final_challenge_ugv1'] * sample_count,
            'timestamp_ns': timestamps_ns,
            'q_x': np.zeros(sample_count),
            'q_y': np.zeros(sample_count),
            'q_z': np.zeros(sample_count),
            'q_w': np.ones(sample_count),
            'ang_vel_x': np.zeros(sample_count),
            'ang_vel_y': np.zeros(sample_count),
            'ang_vel_z': np.zeros(sample_count),
            'lin_acc_x': accel_x(t),
            'lin_acc_y': np.zeros(sample_count),
            'lin_acc_z': np.zeros(sample_count),
        }
    )


def make_quaternion_flip_imu_frame(*, source_rate_hz: int, duration_s: float) -> pd.DataFrame:
    sample_count = int(source_rate_hz * duration_s)
    dt_ns = int(round(1e9 / source_rate_hz))
    timestamps_ns = np.arange(sample_count, dtype=np.int64) * dt_ns
    signs = np.where(np.arange(sample_count) % 2 == 0, 1.0, -1.0)
    return pd.DataFrame(
        {
            'sequence_name': ['final_challenge_ugv1'] * sample_count,
            'timestamp_ns': timestamps_ns,
            'q_x': np.zeros(sample_count),
            'q_y': np.zeros(sample_count),
            'q_z': np.zeros(sample_count),
            'q_w': signs,
            'ang_vel_x': np.zeros(sample_count),
            'ang_vel_y': np.zeros(sample_count),
            'ang_vel_z': np.zeros(sample_count),
            'lin_acc_x': np.zeros(sample_count),
            'lin_acc_y': np.zeros(sample_count),
            'lin_acc_z': np.zeros(sample_count),
        }
    )


def _fft_amplitude(signal: np.ndarray, *, rate_hz: int, frequency_hz: float) -> float:
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / rate_hz)
    spectrum = np.fft.rfft(signal)
    idx = int(np.argmin(np.abs(frequencies - frequency_hz)))
    return float((2.0 / signal.size) * np.abs(spectrum[idx]))


def test_resample_imu_preserves_low_band_motion_and_rejects_aliasing():
    frame = make_synthetic_imu_frame(
        source_rate_hz=200,
        duration_s=2.0,
        accel_x=lambda t: np.sin(2 * np.pi * 5 * t) + 0.35 * np.sin(2 * np.pi * 35 * t),
    )

    output = resample_imu_frame(frame, target_rate_hz=50)

    timestamps = output['timestamp_ns'].to_numpy()
    dt_ns = np.diff(timestamps)
    target_t = (timestamps - timestamps[0]) / 1e9
    expected_low_band = np.sin(2 * np.pi * 5 * target_t)
    accel_x = output['lin_acc_x'].to_numpy()

    # Exact 20 ms spacing is required so later window extraction yields deterministic window sizes and latency.
    assert np.all(dt_ns == 20_000_000)
    # The grid row count must be exact or downstream train/inference window counts silently drift.
    assert len(output) == 100

    rmse = np.sqrt(np.mean((accel_x - expected_low_band) ** 2))
    # Low-band motion must survive resampling or true platform dynamics disappear from model inputs.
    assert rmse < 0.15

    low_band_amp = _fft_amplitude(accel_x, rate_hz=50, frequency_hz=5.0)
    alias_amp = _fft_amplitude(accel_x, rate_hz=50, frequency_hz=15.0)
    # The 35 Hz source term would alias near 15 Hz without anti-alias filtering, so this must stay strongly suppressed.
    assert alias_amp < 0.12 * low_band_amp

    # Finite outputs are required because one non-finite sample can poison derived features and normalizer fitting.
    assert np.isfinite(output.drop(columns=['sequence_name']).to_numpy()).all()


def test_resample_imu_repairs_quaternion_sign_flips_and_renormalizes():
    frame = make_quaternion_flip_imu_frame(source_rate_hz=200, duration_s=1.0)
    output = resample_imu_frame(frame, target_rate_hz=50)

    quat = output[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
    norms = np.linalg.norm(quat, axis=1)
    consecutive_dots = np.sum(quat[1:] * quat[:-1], axis=1)

    # Unit-norm quaternions are required so orientation remains physically valid downstream.
    assert np.allclose(norms, 1.0, atol=1e-4)
    # Non-negative consecutive dot products mean artificial sign inversions were removed before interpolation.
    assert np.all(consecutive_dots >= 0.0)
    # Constant orientation input should remain nearly constant; drift here would be introduced by preprocessing.
    assert np.max(np.abs(quat - quat[0])) < 1e-3
