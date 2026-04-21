from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NORMALIZER_EPSILON = 1e-12


def fit_normalizer(
    feature_tables: dict[str, pd.DataFrame],
    training_sequences: list[str],
    feature_columns: list[str],
    mode: str,
    output_path: Path | None = None,
) -> dict[str, object]:
    if mode not in {'zscore', 'robust'}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of ['zscore', 'robust']")

    missing_sequences = [sequence_name for sequence_name in training_sequences if sequence_name not in feature_tables]
    if missing_sequences:
        raise KeyError(f'Missing feature tables for training sequences: {missing_sequences}')

    train_frame = pd.concat(
        [feature_tables[sequence_name][feature_columns] for sequence_name in training_sequences],
        axis=0,
        ignore_index=True,
    )

    center: dict[str, float] = {}
    scale: dict[str, float] = {}

    for column_name in feature_columns:
        values = train_frame[column_name].to_numpy(dtype=float)

        if mode == 'zscore':
            column_center = float(np.mean(values))
            column_scale = float(np.std(values, ddof=0))
        else:
            column_center = float(np.median(values))
            column_scale = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))

        if abs(column_scale) <= NORMALIZER_EPSILON:
            column_scale = 1.0

        center[column_name] = column_center
        scale[column_name] = column_scale

    payload: dict[str, object] = {
        'mode': mode,
        'fitted_sequence_names': list(training_sequences),
        'feature_columns': list(feature_columns),
        'center': center,
        'scale': scale,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return payload
