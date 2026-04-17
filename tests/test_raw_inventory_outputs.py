from src.raw_inventory import build_raw_manifest
from tests.cleaning_helpers import UGV_SEQUENCE_NAMES


def test_build_raw_manifest_lists_required_assets_for_all_ugv_sequences():
    listing = {
        sequence_name: [
            'imu_data.csv',
            f'{sequence_name}_gt.zip',
            f'{sequence_name}_folder.zip',
            'calibration.yaml',
        ]
        for sequence_name in UGV_SEQUENCE_NAMES
    }

    manifest = build_raw_manifest(listing)

    assert [row['sequence_name'] for row in manifest] == UGV_SEQUENCE_NAMES
    assert all(row['has_imu_csv'] for row in manifest)
    assert all(row['has_gt_zip'] for row in manifest)
    assert all(row['has_folder_zip'] for row in manifest)
    assert all(row['has_calibration'] for row in manifest)
