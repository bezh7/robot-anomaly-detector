import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from src.data.canonicalize_gt import canonicalize_gt_csv
from tests.cleaning_helpers import write_sample_gt_csv


def test_canonicalize_gt_output_repairs_quaternion_sign_flips(tmp_path):
    gt_path = write_sample_gt_csv(
        tmp_path / 'ground_truth_path.csv',
        timestamps=[300, 100, 200],
        sign_flip_index=1,
    )

    output_path = canonicalize_gt_csv(
        sequence_name='urban_challenge_ugv2',
        input_path=gt_path,
        output_dir=tmp_path / 'gt_canonical',
    )
    frame = pd.read_parquet(output_path)

    assert list(frame.columns) == [
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
    assert frame['sequence_name'].nunique() == 1
    assert frame['sequence_name'].iloc[0] == 'urban_challenge_ugv2'
    assert frame['timestamp_ns'].tolist() == [100, 200, 300]
    assert is_integer_dtype(frame['timestamp_ns'])
    assert all(is_float_dtype(frame[column]) for column in frame.columns[2:])
    assert frame.isna().sum().sum() == 0

    quaternions = frame[['q_w_b_x', 'q_w_b_y', 'q_w_b_z', 'q_w_b_w']].to_numpy()
    consecutive_dots = [float(np.dot(previous, current)) for previous, current in zip(quaternions, quaternions[1:])]
    assert all(dot >= 0.0 for dot in consecutive_dots)
