import pandas as pd

from src.canonicalize_gt import canonicalize_gt_csv
from src.canonicalize_imu import canonicalize_imu_csv
from src.trim_overlap import trim_sequence_overlap
from tests.cleaning_helpers import write_sample_gt_csv, write_sample_imu_csv


def test_trim_sequence_overlap_writes_trimmed_outputs_with_expected_bounds(tmp_path):
    imu_input_path = write_sample_imu_csv(
        tmp_path / 'imu_data.csv',
        timestamps=[100, 200, 300, 400, 500],
    )
    gt_input_path = write_sample_gt_csv(
        tmp_path / 'ground_truth_path.csv',
        timestamps=[250, 350, 450, 550],
        sign_flip_index=1,
    )

    imu_output_path = canonicalize_imu_csv('urban_challenge_ugv1', imu_input_path, tmp_path / 'imu_canonical')
    gt_output_path = canonicalize_gt_csv('urban_challenge_ugv1', gt_input_path, tmp_path / 'gt_canonical')

    metadata = trim_sequence_overlap(
        sequence_name='urban_challenge_ugv1',
        imu_path=imu_output_path,
        gt_path=gt_output_path,
        output_dir=tmp_path / 'overlap',
    )

    trimmed_imu = pd.read_parquet(metadata['imu_output_path'])
    trimmed_gt = pd.read_parquet(metadata['gt_output_path'])

    assert metadata['overlap_start_ns'] == 250
    assert metadata['overlap_end_ns'] == 500
    assert metadata['imu_row_count_before'] == 5
    assert metadata['imu_row_count_after'] == 3
    assert metadata['gt_row_count_before'] == 4
    assert metadata['gt_row_count_after'] == 3
    assert trimmed_imu['timestamp_ns'].tolist() == [300, 400, 500]
    assert trimmed_gt['timestamp_ns'].tolist() == [250, 350, 450]
