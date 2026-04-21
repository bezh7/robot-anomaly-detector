from src.modeling.modeling_contract import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_FEATURE_SETS,
    DEFAULT_FOLDS_STAGE_ONE,
    DEFAULT_LEARNING_RATES,
    MODELING_ARTIFACT_DIRNAME,
    build_experiment_id,
)


def test_modeling_contract_exposes_locked_search_space():
    assert DEFAULT_FEATURE_SETS == ("raw", "raw_plus_derived")
    assert DEFAULT_BATCH_SIZES == (32, 64, 128)
    assert DEFAULT_LEARNING_RATES == (3e-4, 1e-3)
    assert DEFAULT_FOLDS_STAGE_ONE == ("fold_2", "fold_4")
    assert MODELING_ARTIFACT_DIRNAME == "modeling"


def test_build_experiment_id_is_stable():
    experiment_id = build_experiment_id(
        architecture="lstm",
        feature_set="raw",
        normalization="zscore",
        batch_size=64,
        learning_rate=1e-3,
        seed=7,
    )
    assert experiment_id == "lstm-raw-zscore-bs64-lr1e-03-seed7"
