from __future__ import annotations

UGV_SEQUENCE_NAMES = [
    'final_challenge_ugv1',
    'final_challenge_ugv2',
    'final_challenge_ugv3',
    'urban_challenge_ugv1',
    'urban_challenge_ugv2',
]

IMU_RAW_FILENAME = 'imu_data.csv'
CALIBRATION_FILENAME = 'calibration.yaml'
GROUND_TRUTH_CSV_FILENAME = 'ground_truth_path.csv'

IMU_CANONICAL_COLUMNS = [
    'sequence_name',
    'timestamp_ns',
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

GT_CANONICAL_COLUMNS = [
    'sequence_name',
    'timestamp_ns',
    'p_w_b_x',
    'p_w_b_y',
    'p_w_b_z',
    'q_w_b_x',
    'q_w_b_y',
    'q_w_b_z',
    'q_w_b_w',
]

GT_QUATERNION_COLUMNS = ['q_w_b_x', 'q_w_b_y', 'q_w_b_z', 'q_w_b_w']
