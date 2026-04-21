import pandas as pd

from src.features.build_window_index import build_window_index


def make_feature_table_with_n_rows(*, sequence_name: str, n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'sequence_name': [sequence_name] * n_rows,
            'timestamp_ns': [index * 20_000_000 for index in range(n_rows)],
            'timestep_index': list(range(n_rows)),
        }
    )


def test_build_window_index_matches_formula_and_never_crosses_boundaries():
    feature_table = make_feature_table_with_n_rows(sequence_name='seq_a', n_rows=400)

    index = build_window_index(
        feature_table=feature_table,
        fold_name='fold_1',
        split_name='train',
        feature_set_name='raw',
        normalization_mode='zscore',
        window_size=150,
        stride=50,
    )

    # Fold/split/feature metadata must be attached to every window for split-safe experiment bookkeeping.
    assert index['fold_name'].eq('fold_1').all()
    assert index['split_name'].eq('train').all()
    assert index['feature_set_name'].eq('raw').all()
    assert index['normalization_mode'].eq('zscore').all()
    # Count must follow the exact rolling-window formula so training volume is predictable.
    assert len(index) == ((400 - 150) // 50) + 1
    # Each window must have the configured fixed length so model tensors are shape-stable.
    assert (index['end_row'] - index['start_row'] + 1).eq(150).all()
    # Window starts should advance by exactly the requested stride to preserve overlap semantics.
    assert index['start_row'].diff().dropna().eq(50).all()
    # Every window must stay inside sequence bounds or downstream slicing will fail at runtime.
    assert index['start_row'].min() >= 0
    assert index['end_row'].max() < 400
    # Timestamp span metadata must reflect exactly the selected rows for temporal attribution.
    assert (index['end_timestamp_ns'] - index['start_timestamp_ns']).eq(149 * 20_000_000).all()
