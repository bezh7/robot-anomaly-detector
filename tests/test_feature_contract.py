from src.features.feature_contract import (
    DEFAULT_INFERENCE_STRIDE,
    DEFAULT_TARGET_RATE_HZ,
    DEFAULT_TRAIN_STRIDE,
    DEFAULT_WINDOW_SIZE,
    DERIVED_IMU_FEATURES,
    FEATURE_SET_COLUMNS,
    FEATURE_TABLE_COLUMNS,
    FEATURE_TABLE_METADATA_COLUMNS,
    GT_CONTEXT_FEATURES,
    NORMALIZATION_MODES,
    RAW_IMU_FEATURES,
)


def test_feature_contract_matches_design_decisions_exactly():
    # Raw feature order defines baseline tensor layout, so this must stay exact.
    assert RAW_IMU_FEATURES == [
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
    # Derived feature order defines the raw+derived ablation layout.
    assert DERIVED_IMU_FEATURES == [
        'gyro_norm',
        'accel_norm',
        'angular_accel_norm',
        'jerk_norm',
        'gyro_rms_local',
        'accel_rms_local',
    ]
    # GT context is tracked separately from model-input features.
    assert GT_CONTEXT_FEATURES == [
        'gt_speed_mps',
        'gt_horizontal_speed_mps',
        'gt_vertical_speed_mps',
        'gt_yaw_rad',
        'gt_yaw_rate_rps',
    ]
    # Metadata columns anchor every feature row back to sequence and time.
    assert FEATURE_TABLE_METADATA_COLUMNS == ['sequence_name', 'timestamp_ns', 'timestep_index']
    # Full feature-table order must remain deterministic for downstream slicing and validation.
    assert FEATURE_TABLE_COLUMNS == FEATURE_TABLE_METADATA_COLUMNS + RAW_IMU_FEATURES + DERIVED_IMU_FEATURES + GT_CONTEXT_FEATURES
    # Feature-set names must stay aligned with the agreed experiment matrix.
    assert list(FEATURE_SET_COLUMNS.keys()) == ['raw', 'raw_plus_derived']
    assert FEATURE_SET_COLUMNS['raw'] == RAW_IMU_FEATURES
    assert FEATURE_SET_COLUMNS['raw_plus_derived'] == RAW_IMU_FEATURES + DERIVED_IMU_FEATURES
    # Normalization modes are intentionally limited to the agreed comparison pair.
    assert NORMALIZATION_MODES == ['zscore', 'robust']
    # Timing defaults encode the approved 50 Hz / 3 s / 1 s / 100 ms setup.
    assert DEFAULT_TARGET_RATE_HZ == 50
    assert DEFAULT_WINDOW_SIZE == 150
    assert DEFAULT_TRAIN_STRIDE == 50
    assert DEFAULT_INFERENCE_STRIDE == 5
