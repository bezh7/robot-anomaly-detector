from __future__ import annotations

import numpy as np


GROUP_SLICES = {
    "quaternion": slice(0, 4),
    "gyro": slice(4, 7),
    "accel": slice(7, 10),
}


def aggregate_top_timesteps(residuals: np.ndarray, top_fraction: float = 0.05) -> float:
    residuals = np.asarray(residuals, dtype=float)
    per_timestep = residuals.mean(axis=1)
    top_k = max(1, int(np.ceil(per_timestep.shape[0] * top_fraction)))
    return float(np.sort(per_timestep)[-top_k:].mean())


def compute_group_scores(residuals: np.ndarray, top_fraction: float = 0.05) -> dict[str, float]:
    residuals = np.asarray(residuals, dtype=float)
    return {
        group_name: aggregate_top_timesteps(residuals[:, group_slice], top_fraction=top_fraction)
        for group_name, group_slice in GROUP_SLICES.items()
    }


def fit_percentile_threshold(scores: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(scores, dtype=float), percentile))


def passes_persistence_rule(history: list[bool], required_consecutive: int = 2) -> bool:
    if len(history) < required_consecutive:
        return False
    return all(history[-required_consecutive:])
