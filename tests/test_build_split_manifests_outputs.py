from src.build_split_manifests import build_dev_split_manifest


def test_build_split_manifest_has_no_leakage_and_exact_phase1_experiments():
    manifest = build_dev_split_manifest(
        dev_sequences=['run_a', 'run_b', 'run_c', 'run_d'],
        feature_sets=['raw', 'raw_plus_derived'],
        normalization_modes=['zscore', 'robust'],
        window_size=150,
    )

    validation_runs = [fold['validation_sequence'] for fold in manifest['folds']]

    # Every dev run must appear exactly once as validation so fold metrics are comparable.
    assert sorted(validation_runs) == ['run_a', 'run_b', 'run_c', 'run_d']

    for fold in manifest['folds']:
        training_runs = set(fold['training_sequences'])
        validation_run = fold['validation_sequence']

        # Train/validation overlap would leak information and inflate validation performance.
        assert validation_run not in training_runs
        # With four dev runs, each fold must train on exactly three to keep fold sizes consistent.
        assert len(training_runs) == 3

    # Phase-1 must stay fixed to these four configs for reproducible ablations and comparisons.
    assert manifest['experiments'] == [
        {'feature_set': 'raw', 'normalization_mode': 'zscore', 'window_size': 150},
        {'feature_set': 'raw', 'normalization_mode': 'robust', 'window_size': 150},
        {'feature_set': 'raw_plus_derived', 'normalization_mode': 'zscore', 'window_size': 150},
        {'feature_set': 'raw_plus_derived', 'normalization_mode': 'robust', 'window_size': 150},
    ]
