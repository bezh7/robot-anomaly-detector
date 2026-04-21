from __future__ import annotations


def build_dev_split_manifest(
    dev_sequences: list[str],
    feature_sets: list[str],
    normalization_modes: list[str],
    window_size: int,
) -> dict[str, list[dict[str, object]]]:
    folds = []
    for validation_sequence in dev_sequences:
        training_sequences = [sequence for sequence in dev_sequences if sequence != validation_sequence]
        folds.append({
            'training_sequences': training_sequences,
            'validation_sequence': validation_sequence,
        })

    experiments = [
        {
            'feature_set': feature_set,
            'normalization_mode': normalization_mode,
            'window_size': window_size,
        }
        for feature_set in feature_sets
        for normalization_mode in normalization_modes
    ]

    return {
        'folds': folds,
        'experiments': experiments,
    }
