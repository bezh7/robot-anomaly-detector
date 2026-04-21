from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.modeling_contract import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_FEATURE_SETS,
    DEFAULT_FOLDS_STAGE_ONE,
    DEFAULT_LEARNING_RATES,
)
from src.modeling.build_modeling_leaderboard import write_experiment_manifest, write_leaderboard
from src.modeling.evaluate_anomaly_model import evaluate_experiment
from src.modeling.train_anomaly_model import train_experiment


def build_stage_one_trials(
    *,
    architectures: tuple[str, ...],
    feature_sets: tuple[str, ...],
    normalizations: tuple[str, ...],
    batch_sizes: tuple[int, ...],
    learning_rates: tuple[float, ...],
    trials_per_architecture: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    all_trial_specs = list(
        itertools.product(feature_sets, normalizations, batch_sizes, learning_rates)
    )
    trials = []
    for architecture in architectures:
        shuffled_specs = list(all_trial_specs)
        rng.shuffle(shuffled_specs)
        if trials_per_architecture <= len(shuffled_specs):
            selected_specs = shuffled_specs[:trials_per_architecture]
        else:
            selected_specs = shuffled_specs[:]
            while len(selected_specs) < trials_per_architecture:
                selected_specs.append(rng.choice(all_trial_specs))
        for feature_set, normalization, batch_size, learning_rate in selected_specs:
            trials.append(
                {
                    "architecture": architecture,
                    "feature_set": feature_set,
                    "normalization": normalization,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                }
            )
    return trials


def run_stage_one_search(
    *,
    artifact_root: Path | str,
    output_root: Path | str,
    architectures: tuple[str, ...],
    feature_sets: tuple[str, ...],
    normalizations: tuple[str, ...],
    batch_sizes: tuple[int, ...],
    learning_rates: tuple[float, ...],
    folds: tuple[str, ...],
    trials_per_architecture: int,
    max_epochs: int,
    seed: int,
) -> list[dict]:
    artifact_root = Path(artifact_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    trials = build_stage_one_trials(
        architectures=architectures,
        feature_sets=feature_sets,
        normalizations=normalizations,
        batch_sizes=batch_sizes,
        learning_rates=learning_rates,
        trials_per_architecture=trials_per_architecture,
        seed=seed,
    )

    experiment_rows = []
    for trial in trials:
        fold_rows = []
        experiment_id = None
        for fold_name in folds:
            train_result = train_experiment(
                artifact_root=artifact_root,
                output_dir=output_root,
                architecture=trial["architecture"],
                feature_set=trial["feature_set"],
                normalization=trial["normalization"],
                fold_name=fold_name,
                batch_size=int(trial["batch_size"]),
                learning_rate=float(trial["learning_rate"]),
                max_epochs=max_epochs,
                seed=seed,
            )
            experiment_id = train_result.experiment_id
            evaluate_result = evaluate_experiment(
                artifact_root=artifact_root,
                output_dir=output_root,
                experiment_id=train_result.experiment_id,
                checkpoint_path=train_result.best_checkpoint_path,
                fold_name=fold_name,
                seed=seed,
            )
            fold_rows.append(
                {
                    "fold_name": fold_name,
                    "clean_alerts_per_min": evaluate_result.clean_summary["clean_alerts_per_min"],
                    "event_detection_rate": evaluate_result.injected_summary["event_detection_rate"],
                    "median_time_to_detect_s": evaluate_result.injected_summary["median_time_to_detect_s"],
                    "group_attribution_top1": evaluate_result.injected_summary["group_attribution_top1"],
                    "channel_top3_hit_rate": evaluate_result.injected_summary["channel_top3_hit_rate"],
                }
            )

        experiment_rows.append(
            {
                "experiment_id": experiment_id,
                "architecture": trial["architecture"],
                "feature_set": trial["feature_set"],
                "normalization": trial["normalization"],
                "batch_size": int(trial["batch_size"]),
                "learning_rate": float(trial["learning_rate"]),
                "folds_evaluated": len(folds),
                "seed_count": 1,
                "clean_alerts_per_min": float(pd.DataFrame(fold_rows)["clean_alerts_per_min"].mean()),
                "event_detection_rate": float(pd.DataFrame(fold_rows)["event_detection_rate"].mean()),
                "median_time_to_detect_s": _aggregate_median_time(pd.DataFrame(fold_rows)["median_time_to_detect_s"]),
                "group_attribution_top1": float(pd.DataFrame(fold_rows)["group_attribution_top1"].mean()),
                "channel_top3_hit_rate": float(pd.DataFrame(fold_rows)["channel_top3_hit_rate"].mean()),
            }
        )

    ranked_rows = write_leaderboard(rows=experiment_rows, output_root=output_root)
    write_experiment_manifest(rows=ranked_rows, trials=trials, output_root=output_root)
    return ranked_rows


def _aggregate_median_time(series: pd.Series) -> float:
    finite = series[np.isfinite(series.to_numpy(dtype=float))]
    if len(finite) == 0:
        return float("inf")
    return float(finite.median())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stage-one anomaly-model search pipeline.")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--architectures", default="lstm,tcn")
    parser.add_argument("--feature-sets", default=",".join(DEFAULT_FEATURE_SETS))
    parser.add_argument("--normalizations", default="zscore,robust")
    parser.add_argument("--batch-sizes", default=",".join(str(value) for value in DEFAULT_BATCH_SIZES))
    parser.add_argument("--learning-rates", default=",".join(f"{value:g}" for value in DEFAULT_LEARNING_RATES))
    parser.add_argument("--folds", default=",".join(DEFAULT_FOLDS_STAGE_ONE))
    parser.add_argument("--trials-per-architecture", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    run_stage_one_search(
        artifact_root=args.artifact_root,
        output_root=args.output_root,
        architectures=_parse_csv(args.architectures),
        feature_sets=_parse_csv(args.feature_sets),
        normalizations=_parse_csv(args.normalizations),
        batch_sizes=tuple(int(value) for value in _parse_csv(args.batch_sizes)),
        learning_rates=tuple(float(value) for value in _parse_csv(args.learning_rates)),
        folds=_parse_csv(args.folds),
        trials_per_architecture=args.trials_per_architecture,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
