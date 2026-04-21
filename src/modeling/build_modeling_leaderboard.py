from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


def rank_experiments(rows: list[dict]) -> list[dict]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["event_detection_rate"]),
            _sortable_float(row["median_time_to_detect_s"]),
            _sortable_float(row["clean_alerts_per_min"]),
            -float(row["group_attribution_top1"]),
        ),
    )
    return ranked


def write_leaderboard(*, rows: list[dict], output_root: Path | str) -> list[dict]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ranked_rows = []
    for rank, row in enumerate(rank_experiments(rows), start=1):
        ranked_rows.append({**row, "overall_rank": rank})

    pd.DataFrame(ranked_rows).to_csv(output_root / "leaderboard.csv", index=False)
    (output_root / "leaderboard.json").write_text(json.dumps(_sanitize_rows(ranked_rows), indent=2))
    return ranked_rows


def write_experiment_manifest(*, rows: list[dict], trials: list[dict], output_root: Path | str) -> None:
    output_root = Path(output_root)
    payload = {
        "trial_count": len(trials),
        "trials": trials,
        "experiments": _sanitize_rows(rows),
    }
    (output_root / "experiment_manifest.json").write_text(json.dumps(payload, indent=2))


def _sanitize_rows(rows: list[dict]) -> list[dict]:
    return [{key: _sanitize_value(value) for key, value in row.items()} for row in rows]


def _sanitize_value(value):
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    return value


def _sortable_float(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("inf")
