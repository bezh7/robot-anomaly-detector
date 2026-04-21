from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.build_feature_tables import build_feature_table
from src.features.build_split_manifests import build_dev_split_manifest
from src.features.build_window_index import build_window_index
from src.features.feature_contract import (
    DEFAULT_WINDOW_SIZE,
    FEATURE_SET_COLUMNS,
    NORMALIZATION_MODES,
)
from src.features.fit_normalizers import fit_normalizer
from src.features.validate_feature_dataset import validate_feature_dataset

RESTORE_CLEAN_ARTIFACTS_COMMAND = (
    'aws s3 sync '
    's3://<bucket>/<artifact-prefix>/clean '
    'artifacts/clean'
)


def _infer_dev_sequences(clean_root: Path) -> list[str]:
    overlap_dir = clean_root / 'overlap'
    return sorted(path.name.removesuffix('_imu.parquet') for path in overlap_dir.glob('*_imu.parquet'))


def build_feature_dataset(
    clean_root: Path,
    output_root: Path,
    *,
    dev_sequences: list[str] | None = None,
) -> Path:
    clean_root = Path(clean_root)
    output_root = Path(output_root)

    if not clean_root.exists():
        raise FileNotFoundError(
            f'Clean artifact layer is required before building feature datasets. '
            f'Missing path: {clean_root}. '
            f'Restore with: {RESTORE_CLEAN_ARTIFACTS_COMMAND}'
        )

    overlap_dir = clean_root / 'overlap'
    if not overlap_dir.exists():
        raise FileNotFoundError(f'Expected overlap artifact directory at {overlap_dir}')

    dev_sequences = list(dev_sequences) if dev_sequences is not None else _infer_dev_sequences(clean_root)
    if not dev_sequences:
        raise ValueError('No development sequences were provided or discovered from clean artifacts')

    feature_tables_dir = output_root / 'feature_tables'
    normalizers_dir = output_root / 'normalizers'
    window_indices_dir = output_root / 'window_indices'
    output_root.mkdir(parents=True, exist_ok=True)
    feature_tables_dir.mkdir(parents=True, exist_ok=True)
    normalizers_dir.mkdir(parents=True, exist_ok=True)
    window_indices_dir.mkdir(parents=True, exist_ok=True)

    feature_table_records: list[dict[str, object]] = []
    feature_tables_by_sequence: dict[str, pd.DataFrame] = {}
    for sequence_name in dev_sequences:
        imu_input_path = overlap_dir / f'{sequence_name}_imu.parquet'
        gt_input_path = overlap_dir / f'{sequence_name}_gt.parquet'
        table_path, manifest_record = build_feature_table(
            sequence_name=sequence_name,
            imu_input_path=imu_input_path,
            gt_input_path=gt_input_path,
            output_dir=feature_tables_dir,
        )
        feature_table_records.append(manifest_record)
        feature_tables_by_sequence[sequence_name] = pd.read_parquet(table_path)

    (output_root / 'feature_table_manifest.json').write_text(
        json.dumps(feature_table_records, indent=2, sort_keys=True)
    )

    split_manifest = build_dev_split_manifest(
        dev_sequences=dev_sequences,
        feature_sets=list(FEATURE_SET_COLUMNS.keys()),
        normalization_modes=NORMALIZATION_MODES,
        window_size=DEFAULT_WINDOW_SIZE,
    )
    (output_root / 'split_manifest.json').write_text(json.dumps(split_manifest, indent=2, sort_keys=True))

    for fold_index, fold in enumerate(split_manifest['folds'], start=1):
        fold_name = f'fold_{fold_index}'
        training_sequences = list(fold['training_sequences'])
        validation_sequence = str(fold['validation_sequence'])
        training_table = pd.concat(
            [feature_tables_by_sequence[name] for name in training_sequences],
            axis=0,
            ignore_index=True,
        )
        validation_table = feature_tables_by_sequence[validation_sequence]

        for experiment in split_manifest['experiments']:
            feature_set_name = str(experiment['feature_set'])
            normalization_mode = str(experiment['normalization_mode'])
            window_size = int(experiment['window_size'])

            fit_normalizer(
                feature_tables=feature_tables_by_sequence,
                training_sequences=training_sequences,
                feature_columns=FEATURE_SET_COLUMNS[feature_set_name],
                mode=normalization_mode,
                output_path=normalizers_dir / f'{fold_name}_{feature_set_name}_{normalization_mode}.json',
            )

            train_index = build_window_index(
                feature_table=training_table,
                fold_name=fold_name,
                split_name='train',
                feature_set_name=feature_set_name,
                normalization_mode=normalization_mode,
                window_size=window_size,
            )
            train_index.to_parquet(
                window_indices_dir / f'{fold_name}_train_{feature_set_name}_{normalization_mode}.parquet',
                index=False,
            )

            validation_index = build_window_index(
                feature_table=validation_table,
                fold_name=fold_name,
                split_name='validation',
                feature_set_name=feature_set_name,
                normalization_mode=normalization_mode,
                window_size=window_size,
            )
            validation_index.to_parquet(
                window_indices_dir / f'{fold_name}_validation_{feature_set_name}_{normalization_mode}.parquet',
                index=False,
            )

    validation_summary = validate_feature_dataset(output_root)
    (output_root / 'build_summary.json').write_text(
        json.dumps(
            {
                'dev_sequences': dev_sequences,
                'window_size': DEFAULT_WINDOW_SIZE,
                'feature_sets': list(FEATURE_SET_COLUMNS.keys()),
                'normalization_modes': NORMALIZATION_MODES,
                'validation_summary': validation_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )

    return output_root
