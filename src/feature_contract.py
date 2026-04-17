from __future__ import annotations

RAW_IMU_FEATURES = [
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

DERIVED_IMU_FEATURES = [
    'gyro_norm',
    'accel_norm',
    'angular_accel_norm',
    'jerk_norm',
    'gyro_rms_local',
    'accel_rms_local',
]

GT_CONTEXT_FEATURES = [
    'gt_speed_mps',
    'gt_horizontal_speed_mps',
    'gt_vertical_speed_mps',
    'gt_yaw_rad',
    'gt_yaw_rate_rps',
]

FEATURE_TABLE_METADATA_COLUMNS = [
    'sequence_name',
    'timestamp_ns',
    'timestep_index',
]

FEATURE_TABLE_COLUMNS = (
    FEATURE_TABLE_METADATA_COLUMNS
    + RAW_IMU_FEATURES
    + DERIVED_IMU_FEATURES
    + GT_CONTEXT_FEATURES
)

FEATURE_SET_COLUMNS = {
    'raw': RAW_IMU_FEATURES,
    'raw_plus_derived': RAW_IMU_FEATURES + DERIVED_IMU_FEATURES,
}

NORMALIZATION_MODES = ['zscore', 'robust']

DEFAULT_TARGET_RATE_HZ = 50
DEFAULT_WINDOW_SIZE = 150
DEFAULT_TRAIN_STRIDE = 50
DEFAULT_INFERENCE_STRIDE = 5
