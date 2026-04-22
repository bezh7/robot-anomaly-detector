from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.feature_contract import FEATURE_SET_COLUMNS


@dataclass(frozen=True)
class FoldInputs:
    fold_name: str
    train_sequences: tuple[str, ...]
    validation_sequence: str
    windows_by_split: dict[str, list[dict]]
    feature_tables: dict[str, pd.DataFrame]
    normalizer: dict
    feature_columns: tuple[str, ...]


def load_fold_inputs(
    artifact_root: Path | str,
    *,
    fold_name: str,
    feature_set: str,
    normalization: str,
) -> FoldInputs:
    artifact_root = Path(artifact_root)

    split_manifest = json.loads((artifact_root / "split_manifest.json").read_text())
    fold_spec = _select_fold_spec(split_manifest, fold_name=fold_name)
    feature_tables = load_feature_tables(artifact_root)

    windows_by_split = {}
    for split_name in ("train", "validation"):
        window_path = artifact_root / "window_indices" / f"{fold_name}_{split_name}_{feature_set}_{normalization}.parquet"
        windows_by_split[split_name] = pd.read_parquet(window_path).to_dict(orient="records")

    normalizer_path = artifact_root / "normalizers" / f"{fold_name}_{feature_set}_{normalization}.json"
    normalizer = json.loads(normalizer_path.read_text())
    feature_columns = tuple(normalizer["feature_columns"])

    return FoldInputs(
        fold_name=fold_name,
        train_sequences=tuple(fold_spec["training_sequences"]),
        validation_sequence=fold_spec["validation_sequence"],
        windows_by_split=windows_by_split,
        feature_tables=feature_tables,
        normalizer=normalizer,
        feature_columns=feature_columns,
    )


def load_feature_tables(artifact_root: Path | str) -> dict[str, pd.DataFrame]:
    artifact_root = Path(artifact_root)
    feature_table_manifest = json.loads((artifact_root / "feature_table_manifest.json").read_text())
    return {
        entry["sequence_name"]: pd.read_parquet(_resolve_feature_table_path(artifact_root, entry))
        for entry in feature_table_manifest
    }


def extract_window_array(
    *,
    frame: pd.DataFrame,
    row: dict,
    feature_columns: tuple[str, ...],
    normalizer: dict,
) -> np.ndarray:
    window = frame.iloc[int(row["start_row"]) : int(row["end_row"]) + 1][list(feature_columns)]
    window_array = window.to_numpy(dtype=np.float32, copy=True)
    return apply_normalizer(window_array, feature_columns=feature_columns, normalizer=normalizer)


def apply_normalizer(
    window_array: np.ndarray,
    *,
    feature_columns: tuple[str, ...],
    normalizer: dict,
) -> np.ndarray:
    center_map = normalizer.get("center") or normalizer.get("means") or normalizer.get("medians")
    scale_map = normalizer.get("scale") or normalizer.get("stds") or normalizer.get("iqrs")
    if center_map is None or scale_map is None:
        raise ValueError("normalizer payload must include center/scale values")

    center = np.asarray([float(center_map[column]) for column in feature_columns], dtype=np.float32)
    scale = np.asarray([float(scale_map[column]) for column in feature_columns], dtype=np.float32)
    scale = np.where(np.abs(scale) <= 1e-12, 1.0, scale)
    return (window_array - center) / scale


def build_feature_columns(feature_set: str) -> tuple[str, ...]:
    try:
        return tuple(FEATURE_SET_COLUMNS[feature_set])
    except KeyError as exc:
        raise ValueError(f"unsupported feature set: {feature_set}") from exc


def _select_fold_spec(split_manifest: dict, *, fold_name: str) -> dict:
    fold_index = int(fold_name.removeprefix("fold_")) - 1
    try:
        return split_manifest["folds"][fold_index]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"unknown fold name: {fold_name}") from exc


def _resolve_feature_table_path(artifact_root: Path, entry: dict) -> Path:
    table_path = Path(entry["feature_table_path"])
    if table_path.is_absolute():
        return table_path
    candidate = artifact_root / table_path
    if candidate.exists():
        return candidate
    if "feature_tables" in table_path.parts:
        feature_tables_index = table_path.parts.index("feature_tables")
        normalized = artifact_root / Path(*table_path.parts[feature_tables_index:])
        if normalized.exists():
            return normalized
    return candidate


class ModelingWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        windows: list[dict],
        feature_tables: dict[str, pd.DataFrame],
        feature_columns: tuple[str, ...],
        normalizer: dict,
    ) -> None:
        self._windows = windows
        self._feature_tables = feature_tables
        self._feature_columns = feature_columns
        self._normalizer = normalizer

    @classmethod
    def from_fold_inputs(cls, fold_inputs: FoldInputs, *, split: str) -> "ModelingWindowDataset":
        return cls(
            windows=fold_inputs.windows_by_split[split],
            feature_tables=fold_inputs.feature_tables,
            feature_columns=fold_inputs.feature_columns,
            normalizer=fold_inputs.normalizer,
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict:
        row = self._windows[index]
        frame = self._feature_tables[row["sequence_name"]]
        return {
            "inputs": torch.from_numpy(
                extract_window_array(
                    frame=frame,
                    row=row,
                    feature_columns=self._feature_columns,
                    normalizer=self._normalizer,
                )
            ),
            "metadata": row,
        }
