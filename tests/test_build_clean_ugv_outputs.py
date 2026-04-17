import json

import pandas as pd

from src.build_clean_ugv_dataset import build_clean_dataset_from_local_root
from src.validate_clean_ugv_dataset import validate_clean_dataset
from tests.cleaning_helpers import UGV_SEQUENCE_NAMES, build_local_ugv_raw_root


def test_build_clean_dataset_writes_all_expected_outputs(tmp_path):
    raw_root = build_local_ugv_raw_root(tmp_path / 'raw_root')
    output_dir = tmp_path / 'artifacts' / 'clean'

    summary = build_clean_dataset_from_local_root(raw_root=raw_root, output_dir=output_dir)
    validation = validate_clean_dataset(output_dir)

    raw_manifest = json.loads((output_dir / 'raw_manifest.json').read_text())
    overlap_manifest = json.loads((output_dir / 'overlap_manifest.json').read_text())

    assert summary['sequence_names'] == UGV_SEQUENCE_NAMES
    assert [row['sequence_name'] for row in raw_manifest] == UGV_SEQUENCE_NAMES
    assert [row['sequence_name'] for row in overlap_manifest] == UGV_SEQUENCE_NAMES
    assert validation['sequence_count'] == 5

    for sequence_name in UGV_SEQUENCE_NAMES:
        imu_canonical_path = output_dir / 'imu_canonical' / f'{sequence_name}.parquet'
        gt_canonical_path = output_dir / 'gt_canonical' / f'{sequence_name}.parquet'
        overlap_imu_path = output_dir / 'overlap' / f'{sequence_name}_imu.parquet'
        overlap_gt_path = output_dir / 'overlap' / f'{sequence_name}_gt.parquet'

        assert imu_canonical_path.exists()
        assert gt_canonical_path.exists()
        assert overlap_imu_path.exists()
        assert overlap_gt_path.exists()

        imu_frame = pd.read_parquet(overlap_imu_path)
        gt_frame = pd.read_parquet(overlap_gt_path)
        assert imu_frame['sequence_name'].nunique() == 1
        assert gt_frame['sequence_name'].nunique() == 1
        assert imu_frame['sequence_name'].iloc[0] == sequence_name
        assert gt_frame['sequence_name'].iloc[0] == sequence_name


def test_build_clean_dataset_preserves_supplied_raw_manifest(tmp_path):
    raw_root = build_local_ugv_raw_root(tmp_path / 'raw_root')
    for sequence_name in UGV_SEQUENCE_NAMES:
        (raw_root / sequence_name / f'{sequence_name}_folder.zip').unlink()

    supplied_manifest = [
        {
            'sequence_name': sequence_name,
            'has_imu_csv': True,
            'has_gt_zip': True,
            'has_folder_zip': True,
            'has_calibration': True,
            'file_names': [
                'imu_data.csv',
                f'{sequence_name}_gt.zip',
                f'{sequence_name}_folder.zip',
                'calibration.yaml',
            ],
        }
        for sequence_name in UGV_SEQUENCE_NAMES
    ]

    output_dir = tmp_path / 'artifacts' / 'clean'
    summary = build_clean_dataset_from_local_root(
        raw_root=raw_root,
        output_dir=output_dir,
        raw_manifest=supplied_manifest,
    )

    raw_manifest = json.loads((output_dir / 'raw_manifest.json').read_text())

    assert summary['sequence_names'] == UGV_SEQUENCE_NAMES
    assert all(row['has_folder_zip'] for row in raw_manifest)
