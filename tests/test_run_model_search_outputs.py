from pathlib import Path

import pandas as pd

from src.modeling.build_modeling_leaderboard import rank_experiments
from src.modeling.run_model_search import build_stage_one_trials, run_stage_one_search
from tests.modeling_helpers import write_minimal_feature_artifacts


def test_build_stage_one_trials_respects_locked_search_space():
    trials = build_stage_one_trials(
        architectures=("lstm", "tcn"),
        feature_sets=("raw", "raw_plus_derived"),
        normalizations=("zscore", "robust"),
        batch_sizes=(32, 64, 128),
        learning_rates=(3e-4, 1e-3),
        trials_per_architecture=8,
        seed=7,
    )

    assert len([trial for trial in trials if trial["architecture"] == "lstm"]) == 8
    assert len([trial for trial in trials if trial["architecture"] == "tcn"]) == 8


def test_rank_experiments_uses_lexicographic_metric_priority():
    ranked = rank_experiments(
        [
            {
                "experiment_id": "a",
                "event_detection_rate": 0.9,
                "median_time_to_detect_s": 0.8,
                "clean_alerts_per_min": 0.2,
                "group_attribution_top1": 0.7,
            },
            {
                "experiment_id": "b",
                "event_detection_rate": 0.8,
                "median_time_to_detect_s": 0.1,
                "clean_alerts_per_min": 0.1,
                "group_attribution_top1": 0.9,
            },
        ]
    )

    assert ranked[0]["experiment_id"] == "a"


def test_run_stage_one_search_writes_leaderboard_outputs(tmp_path: Path):
    artifact_root = write_minimal_feature_artifacts(tmp_path)
    output_root = tmp_path / "modeling"

    run_stage_one_search(
        artifact_root=artifact_root,
        output_root=output_root,
        architectures=("lstm",),
        feature_sets=("raw",),
        normalizations=("zscore",),
        batch_sizes=(32,),
        learning_rates=(1e-3,),
        folds=("fold_2",),
        trials_per_architecture=1,
        max_epochs=1,
        seed=7,
    )

    leaderboard_csv = output_root / "leaderboard.csv"
    leaderboard_json = output_root / "leaderboard.json"
    experiment_manifest = output_root / "experiment_manifest.json"

    assert leaderboard_csv.exists()
    assert leaderboard_json.exists()
    assert experiment_manifest.exists()

    leaderboard = pd.read_csv(leaderboard_csv)
    assert len(leaderboard) == 1
    assert "overall_rank" in leaderboard.columns
