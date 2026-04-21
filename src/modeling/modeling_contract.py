MODELING_ARTIFACT_DIRNAME = "modeling"

DEFAULT_FEATURE_SETS = ("raw", "raw_plus_derived")
DEFAULT_BATCH_SIZES = (32, 64, 128)
DEFAULT_LEARNING_RATES = (3e-4, 1e-3)
DEFAULT_FOLDS_STAGE_ONE = ("fold_2", "fold_4")


def build_experiment_id(
    *,
    architecture: str,
    feature_set: str,
    normalization: str,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> str:
    return (
        f"{architecture}-{feature_set}-{normalization}-"
        f"bs{batch_size}-lr{learning_rate:.0e}-seed{seed}"
    )
