import numpy as np

from src.modeling.model_scoring import (
    aggregate_top_timesteps,
    compute_group_scores,
    fit_percentile_threshold,
    passes_persistence_rule,
)


def test_aggregate_top_timesteps_uses_top_five_percent():
    residuals = np.zeros((150, 10), dtype=float)
    residuals[10:18] = 16.0

    score = aggregate_top_timesteps(residuals, top_fraction=0.05)
    assert score == 16.0


def test_fit_percentile_threshold_and_persistence_rule():
    threshold = fit_percentile_threshold([0.1, 0.2, 0.3, 0.4], percentile=75)
    assert threshold == 0.325
    assert passes_persistence_rule([False, True, True], required_consecutive=2) is True


def test_compute_group_scores_respects_channel_groups():
    residuals = np.zeros((150, 10), dtype=float)
    residuals[20:28, 4:7] = 9.0

    scores = compute_group_scores(residuals, top_fraction=0.05)

    assert scores["gyro"] == 9.0
    assert scores["gyro"] > scores["quaternion"]
    assert scores["gyro"] > scores["accel"]
