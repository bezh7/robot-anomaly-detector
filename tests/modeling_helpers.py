from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_feature_dataset import build_feature_dataset
from src.features.build_window_index import WINDOW_INDEX_COLUMNS
from src.features.feature_contract import (
    DERIVED_IMU_FEATURES,
    FEATURE_SET_COLUMNS,
    FEATURE_TABLE_COLUMNS,
    GT_CONTEXT_FEATURES,
    RAW_IMU_FEATURES,
)


def sample_clean_root(tmp_path: Path) -> Path:
    clean_root = tmp_path / "clean"
    overlap_dir = clean_root / "overlap"
    overlap_dir.mkdir(parents=True, exist_ok=True)

    for index, sequence_name in enumerate(["seq_a", "seq_b", "seq_c", "seq_d"]):
        _write_clean_imu_parquet(
            overlap_dir / f"{sequence_name}_imu.parquet",
            sequence_name=sequence_name,
            phase_shift=0.1 * index,
        )
        _write_clean_gt_parquet(
            overlap_dir / f"{sequence_name}_gt.parquet",
            sequence_name=sequence_name,
        )

    return clean_root


def write_builder_feature_artifacts(root: Path) -> Path:
    return build_feature_dataset(
        clean_root=sample_clean_root(root),
        output_root=root / "features",
        dev_sequences=["seq_a", "seq_b", "seq_c", "seq_d"],
    )


def write_minimal_feature_artifacts(root: Path, *, sequence_rows: int = 220) -> Path:
    artifact_root = root / "features"
    feature_tables_dir = artifact_root / "feature_tables"
    normalizers_dir = artifact_root / "normalizers"
    window_indices_dir = artifact_root / "window_indices"
    feature_tables_dir.mkdir(parents=True, exist_ok=True)
    normalizers_dir.mkdir(parents=True, exist_ok=True)
    window_indices_dir.mkdir(parents=True, exist_ok=True)

    sequences = {
        "final_challenge_ugv1": _build_sequence_frame(sequence_rows, offset=0.0, sequence_name="final_challenge_ugv1"),
        "final_challenge_ugv3": _build_sequence_frame(sequence_rows, offset=10.0, sequence_name="final_challenge_ugv3"),
        "urban_challenge_ugv1": _build_sequence_frame(sequence_rows, offset=20.0, sequence_name="urban_challenge_ugv1"),
        "urban_challenge_ugv2": _build_sequence_frame(sequence_rows, offset=30.0, sequence_name="urban_challenge_ugv2"),
    }

    feature_table_manifest = []
    for sequence_name, frame in sequences.items():
        feature_table_path = feature_tables_dir / f"{sequence_name}.parquet"
        frame.to_parquet(feature_table_path, index=False)
        feature_table_manifest.append(
            {
                "sequence_name": sequence_name,
                "feature_table_path": f"feature_tables/{sequence_name}.parquet",
                "row_count": len(frame),
                "target_rate_hz": 50,
                "timestamp_start_ns": int(frame["timestamp_ns"].iloc[0]),
                "timestamp_end_ns": int(frame["timestamp_ns"].iloc[-1]),
                "imu_input_path": f"clean/overlap/{sequence_name}_imu.parquet",
                "gt_input_path": f"clean/overlap/{sequence_name}_gt.parquet",
            }
        )
    (artifact_root / "feature_table_manifest.json").write_text(
        json.dumps(feature_table_manifest, indent=2, sort_keys=True)
    )

    split_manifest = {
        "folds": [
            {
                "training_sequences": ["final_challenge_ugv3", "urban_challenge_ugv1", "urban_challenge_ugv2"],
                "validation_sequence": "final_challenge_ugv1",
            },
            {
                "training_sequences": ["final_challenge_ugv1", "urban_challenge_ugv1", "urban_challenge_ugv2"],
                "validation_sequence": "final_challenge_ugv3",
            },
            {
                "training_sequences": ["final_challenge_ugv1", "final_challenge_ugv3", "urban_challenge_ugv2"],
                "validation_sequence": "urban_challenge_ugv1",
            },
            {
                "training_sequences": ["final_challenge_ugv1", "final_challenge_ugv3", "urban_challenge_ugv1"],
                "validation_sequence": "urban_challenge_ugv2",
            },
        ],
        "experiments": [
            {"feature_set": feature_set, "normalization_mode": normalization_mode, "window_size": 150}
            for feature_set in FEATURE_SET_COLUMNS
            for normalization_mode in ("zscore", "robust")
        ],
    }
    (artifact_root / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True))

    for fold_index, fold in enumerate(split_manifest["folds"], start=1):
        fold_name = f"fold_{fold_index}"
        for feature_set_name, feature_columns in FEATURE_SET_COLUMNS.items():
            for normalization_mode in ("zscore", "robust"):
                normalizer_payload = {
                    "mode": normalization_mode,
                    "fitted_sequence_names": list(fold["training_sequences"]),
                    "feature_columns": list(feature_columns),
                    "center": {column: 0.0 for column in feature_columns},
                    "scale": {column: 1.0 for column in feature_columns},
                }
                (normalizers_dir / f"{fold_name}_{feature_set_name}_{normalization_mode}.json").write_text(
                    json.dumps(normalizer_payload, indent=2, sort_keys=True)
                )

                training_rows = []
                for sequence_name in fold["training_sequences"]:
                    training_rows.append(
                        {
                            "fold_name": fold_name,
                            "split_name": "train",
                            "feature_set_name": feature_set_name,
                            "normalization_mode": normalization_mode,
                            "sequence_name": sequence_name,
                            "window_size": 150,
                            "stride": 50,
                            "start_row": 0,
                            "end_row": 149,
                            "start_timestamp_ns": 0,
                            "end_timestamp_ns": 149 * 20_000_000,
                        }
                    )
                pd.DataFrame(training_rows, columns=WINDOW_INDEX_COLUMNS).to_parquet(
                    window_indices_dir / f"{fold_name}_train_{feature_set_name}_{normalization_mode}.parquet",
                    index=False,
                )

                validation_rows = [
                    {
                        "fold_name": fold_name,
                        "split_name": "validation",
                        "feature_set_name": feature_set_name,
                        "normalization_mode": normalization_mode,
                        "sequence_name": fold["validation_sequence"],
                        "window_size": 150,
                        "stride": 5,
                        "start_row": 0,
                        "end_row": 149,
                        "start_timestamp_ns": 0,
                        "end_timestamp_ns": 149 * 20_000_000,
                    }
                ]
                pd.DataFrame(validation_rows, columns=WINDOW_INDEX_COLUMNS).to_parquet(
                    window_indices_dir / f"{fold_name}_validation_{feature_set_name}_{normalization_mode}.parquet",
                    index=False,
                )

    return artifact_root


def _build_sequence_frame(row_count: int, *, offset: float, sequence_name: str) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "sequence_name": [sequence_name] * row_count,
            "timestamp_ns": [20_000_000 * index for index in range(row_count)],
            "timestep_index": list(range(row_count)),
        }
    )
    for column_index, column_name in enumerate(RAW_IMU_FEATURES + DERIVED_IMU_FEATURES + GT_CONTEXT_FEATURES):
        base[column_name] = [offset + column_index + (row / 100.0) for row in range(row_count)]
    return base.loc[:, FEATURE_TABLE_COLUMNS]


def _write_clean_imu_parquet(path: Path, *, sequence_name: str, phase_shift: float) -> None:
    dt_s = 0.005
    timestamps_ns = np.arange(0, int(4.0 / dt_s) + 1, dtype=np.int64) * int(dt_s * 1e9)
    time_s = timestamps_ns.astype(float) / 1e9

    frame = pd.DataFrame(
        {
            "sequence_name": sequence_name,
            "timestamp_ns": timestamps_ns,
            "q_x": np.zeros_like(time_s),
            "q_y": np.zeros_like(time_s),
            "q_z": np.zeros_like(time_s),
            "q_w": np.ones_like(time_s),
            "ang_vel_x": np.sin(2.0 * math.pi * (1.0 + phase_shift) * time_s),
            "ang_vel_y": np.cos(2.0 * math.pi * (1.5 + phase_shift) * time_s),
            "ang_vel_z": np.sin(2.0 * math.pi * (0.5 + phase_shift) * time_s),
            "lin_acc_x": 9.81 + 0.2 * np.sin(2.0 * math.pi * (3.0 + phase_shift) * time_s),
            "lin_acc_y": 0.1 * np.cos(2.0 * math.pi * (2.0 + phase_shift) * time_s),
            "lin_acc_z": 0.3 * np.sin(2.0 * math.pi * (1.0 + phase_shift) * time_s),
        }
    )
    frame.to_parquet(path, index=False)


def _write_clean_gt_parquet(path: Path, *, sequence_name: str) -> None:
    timestamps_ns = np.arange(0, 21, dtype=np.int64) * 200_000_000
    time_s = timestamps_ns.astype(float) / 1e9
    yaw = 0.5 * time_s

    frame = pd.DataFrame(
        {
            "sequence_name": sequence_name,
            "timestamp_ns": timestamps_ns,
            "p_w_b_x": 2.0 * time_s,
            "p_w_b_y": np.zeros_like(time_s),
            "p_w_b_z": np.zeros_like(time_s),
            "q_w_b_x": np.zeros_like(time_s),
            "q_w_b_y": np.zeros_like(time_s),
            "q_w_b_z": np.sin(yaw / 2.0),
            "q_w_b_w": np.cos(yaw / 2.0),
        }
    )
    frame.to_parquet(path, index=False)
