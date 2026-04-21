import json

import numpy as np
import pandas as pd
import pytest

from src.features.fit_normalizers import fit_normalizer


def sample_feature_tables_by_sequence() -> dict[str, pd.DataFrame]:
    return {
        'seq_a': pd.DataFrame({'sequence_name': ['seq_a', 'seq_a'], 'q_x': [0.0, 2.0], 'lin_acc_x': [1.0, 3.0]}),
        'seq_b': pd.DataFrame({'sequence_name': ['seq_b', 'seq_b'], 'q_x': [4.0, 6.0], 'lin_acc_x': [5.0, 7.0]}),
        'seq_val': pd.DataFrame(
            {'sequence_name': ['seq_val', 'seq_val'], 'q_x': [100.0, 200.0], 'lin_acc_x': [1000.0, 2000.0]}
        ),
    }


def expected_train_values(
    feature_tables: dict[str, pd.DataFrame],
    training_sequences: list[str],
    column_name: str,
) -> np.ndarray:
    return np.concatenate([feature_tables[sequence_name][column_name].to_numpy(dtype=float) for sequence_name in training_sequences])


def expected_train_mean(
    feature_tables: dict[str, pd.DataFrame],
    training_sequences: list[str],
    column_name: str,
) -> float:
    return float(np.mean(expected_train_values(feature_tables, training_sequences, column_name)))


def expected_train_std(
    feature_tables: dict[str, pd.DataFrame],
    training_sequences: list[str],
    column_name: str,
) -> float:
    return float(np.std(expected_train_values(feature_tables, training_sequences, column_name), ddof=0))


def expected_train_median(
    feature_tables: dict[str, pd.DataFrame],
    training_sequences: list[str],
    column_name: str,
) -> float:
    return float(np.median(expected_train_values(feature_tables, training_sequences, column_name)))


def expected_train_iqr(
    feature_tables: dict[str, pd.DataFrame],
    training_sequences: list[str],
    column_name: str,
) -> float:
    train_values = expected_train_values(feature_tables, training_sequences, column_name)
    return float(np.quantile(train_values, 0.75) - np.quantile(train_values, 0.25))


def test_fit_normalizer_uses_only_training_runs_and_matches_expected_statistics(tmp_path):
    feature_tables = sample_feature_tables_by_sequence()

    zscore_output_path = tmp_path / 'zscore_normalizer.json'
    zscore = fit_normalizer(
        feature_tables=feature_tables,
        training_sequences=['seq_a', 'seq_b'],
        feature_columns=['q_x', 'lin_acc_x'],
        mode='zscore',
        output_path=zscore_output_path,
    )
    robust = fit_normalizer(
        feature_tables=feature_tables,
        training_sequences=['seq_a', 'seq_b'],
        feature_columns=['q_x', 'lin_acc_x'],
        mode='robust',
    )

    # Train-only provenance must be explicit so split leakage is auditable in experiment reviews.
    assert zscore['fitted_sequence_names'] == ['seq_a', 'seq_b']
    # Persisted payload must match the returned payload so downstream loaders see identical normalization math.
    assert json.loads(zscore_output_path.read_text()) == zscore
    # Column ordering must stay exact so tensor feature positions map to the right normalization parameters.
    assert zscore['feature_columns'] == ['q_x', 'lin_acc_x']

    # Mean/std math must be exact for deterministic fold-level z-score normalization.
    assert zscore['center']['q_x'] == pytest.approx(expected_train_mean(feature_tables, ['seq_a', 'seq_b'], 'q_x'))
    assert zscore['center']['lin_acc_x'] == pytest.approx(expected_train_mean(feature_tables, ['seq_a', 'seq_b'], 'lin_acc_x'))
    assert zscore['scale']['q_x'] == pytest.approx(expected_train_std(feature_tables, ['seq_a', 'seq_b'], 'q_x'))
    assert zscore['scale']['lin_acc_x'] == pytest.approx(expected_train_std(feature_tables, ['seq_a', 'seq_b'], 'lin_acc_x'))

    # Median/IQR math must be exact so robust normalization behavior is reproducible and comparable to z-score.
    assert robust['center']['q_x'] == pytest.approx(expected_train_median(feature_tables, ['seq_a', 'seq_b'], 'q_x'))
    assert robust['center']['lin_acc_x'] == pytest.approx(expected_train_median(feature_tables, ['seq_a', 'seq_b'], 'lin_acc_x'))
    assert robust['scale']['q_x'] == pytest.approx(expected_train_iqr(feature_tables, ['seq_a', 'seq_b'], 'q_x'))
    assert robust['scale']['lin_acc_x'] == pytest.approx(expected_train_iqr(feature_tables, ['seq_a', 'seq_b'], 'lin_acc_x'))

    # Validation-only rows must not affect training-fitted centers, or fold metrics are silently inflated.
    assert zscore['center']['q_x'] != pytest.approx(expected_train_mean(feature_tables, ['seq_a', 'seq_b', 'seq_val'], 'q_x'))
