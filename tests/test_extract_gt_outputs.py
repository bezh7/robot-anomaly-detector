import pandas as pd

from src.data.extract_gt import extract_ground_truth_csv
from tests.cleaning_helpers import write_sample_gt_zip


def test_extract_ground_truth_path_writes_expected_csv(tmp_path):
    gt_zip_path = write_sample_gt_zip(
        tmp_path / 'urban_challenge_ugv1_gt.zip',
        timestamps=[100, 200, 300],
        sign_flip_index=1,
    )

    output_path = extract_ground_truth_csv(gt_zip_path, tmp_path / 'gt_raw')
    frame = pd.read_csv(output_path)

    assert output_path.exists()
    assert list(frame.columns) == [
        'timestamp',
        'p_w_b_x',
        'p_w_b_y',
        'p_w_b_z',
        'q_w_b_x',
        'q_w_b_y',
        'q_w_b_z',
        'q_w_b_w',
    ]
    assert len(frame) == 3
