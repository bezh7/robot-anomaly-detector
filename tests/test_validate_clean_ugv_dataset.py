from src.build_clean_ugv_dataset import build_clean_dataset_from_local_root
from src.validate_clean_ugv_dataset import validate_clean_dataset
from tests.cleaning_helpers import build_local_ugv_raw_root


def test_validate_clean_dataset_confirms_expected_clean_artifacts(tmp_path):
    raw_root = build_local_ugv_raw_root(tmp_path / 'raw_root')
    output_dir = tmp_path / 'artifacts' / 'clean'
    build_clean_dataset_from_local_root(raw_root=raw_root, output_dir=output_dir)

    summary = validate_clean_dataset(output_dir)

    assert summary['sequence_count'] == 5
    assert summary['imu_canonical_count'] == 5
    assert summary['gt_canonical_count'] == 5
    assert summary['overlap_pair_count'] == 5
