import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from src.data.canonicalize_imu import canonicalize_imu_csv
from tests.cleaning_helpers import write_sample_imu_csv


def test_canonicalize_imu_output_has_expected_columns_and_clean_timestamps(tmp_path):
    imu_path = write_sample_imu_csv(
        tmp_path / 'imu_data.csv',
        timestamps=[300, 100, 200],
    )

    output_path = canonicalize_imu_csv(
        sequence_name='final_challenge_ugv1',
        input_path=imu_path,
        output_dir=tmp_path / 'imu_canonical',
    )
    frame = pd.read_parquet(output_path)

    assert list(frame.columns) == [
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
    assert frame['sequence_name'].nunique() == 1
    assert frame['sequence_name'].iloc[0] == 'final_challenge_ugv1'
    assert frame['timestamp_ns'].tolist() == [100, 200, 300]
    assert is_integer_dtype(frame['timestamp_ns'])
    assert all(is_float_dtype(frame[column]) for column in frame.columns[2:])
    assert frame.isna().sum().sum() == 0
