from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.modeling.anomaly_injection import DURATIONS, build_injected_sequence
from src.modeling.model_scoring import (
    GROUP_SLICES,
    aggregate_top_timesteps,
    compute_group_scores,
    fit_percentile_threshold,
    passes_persistence_rule,
)
from src.modeling.modeling_dataset import apply_normalizer, load_fold_inputs
from src.modeling.train_anomaly_model import load_model_from_checkpoint


@dataclass(frozen=True)
class EvaluateExperimentResult:
    clean_metrics_path: Path
    injected_metrics_path: Path
    anomaly_breakdown_path: Path
    clean_summary: dict[str, float]
    injected_summary: dict[str, float]


def evaluate_experiment(
    *,
    artifact_root: Path | str,
    output_dir: Path | str,
    experiment_id: str,
    checkpoint_path: Path | str,
    fold_name: str,
    seed: int,
    window_size: int = 150,
    stride: int = 5,
    threshold_percentile: float = 99.0,
    required_consecutive: int = 2,
    grace_period_s: float = 0.5,
    device: str = "cpu",
) -> EvaluateExperimentResult:
    artifact_root = Path(artifact_root)
    output_dir = Path(output_dir)
    run_dir = output_dir / experiment_id / fold_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((run_dir / "config.json").read_text())
    fold_inputs = load_fold_inputs(
        artifact_root,
        fold_name=fold_name,
        feature_set=config["feature_set"],
        normalization=config["normalization"],
    )
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    feature_columns = tuple(checkpoint["feature_columns"])
    validation_frame = fold_inputs.feature_tables[fold_inputs.validation_sequence]

    clean_scores = _score_replay_sequence(
        frame=validation_frame,
        feature_columns=feature_columns,
        normalizer=fold_inputs.normalizer,
        model=model,
        window_size=window_size,
        stride=stride,
        device=device,
    )
    threshold = fit_percentile_threshold([row["score"] for row in clean_scores], threshold_percentile)
    clean_summary = _summarize_clean_scores(
        clean_scores,
        threshold=threshold,
        frame_length=len(validation_frame),
        required_consecutive=required_consecutive,
        stride=stride,
    )

    injection_context = _build_injection_context(validation_frame)
    breakdown_rows = []
    for anomaly_index, anomaly_type in enumerate(DURATIONS):
        start_index = _select_injection_start(
            frame_length=len(validation_frame),
            window_size=window_size,
            duration=DURATIONS[anomaly_type],
        )
        injected = build_injected_sequence(
            validation_frame,
            context=injection_context,
            anomaly_type=anomaly_type,
            severity="medium",
            start_index=start_index,
            seed=seed + anomaly_index,
        )
        injected_scores = _score_replay_sequence(
            frame=injected.frame,
            feature_columns=feature_columns,
            normalizer=fold_inputs.normalizer,
            model=model,
            window_size=window_size,
            stride=stride,
            device=device,
        )
        breakdown_rows.append(
            _evaluate_injected_event(
                score_rows=injected_scores,
                metadata=injected.metadata,
                threshold=threshold,
                required_consecutive=required_consecutive,
                grace_period_s=grace_period_s,
            )
        )

    anomaly_breakdown = pd.DataFrame(breakdown_rows)
    injected_summary = _summarize_injected_breakdown(anomaly_breakdown)

    clean_metrics_path = run_dir / "clean_replay_metrics.csv"
    injected_metrics_path = run_dir / "injected_replay_metrics.csv"
    anomaly_breakdown_path = run_dir / "anomaly_breakdown.csv"

    pd.DataFrame([clean_summary]).to_csv(clean_metrics_path, index=False)
    pd.DataFrame([injected_summary]).to_csv(injected_metrics_path, index=False)
    anomaly_breakdown.to_csv(anomaly_breakdown_path, index=False)

    return EvaluateExperimentResult(
        clean_metrics_path=clean_metrics_path,
        injected_metrics_path=injected_metrics_path,
        anomaly_breakdown_path=anomaly_breakdown_path,
        clean_summary=clean_summary,
        injected_summary=injected_summary,
    )


def _score_replay_sequence(
    *,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    normalizer: dict,
    model,
    window_size: int,
    stride: int,
    device: str,
) -> list[dict[str, object]]:
    score_rows = []
    if len(frame) < window_size:
        return score_rows

    model.eval()
    with torch.no_grad():
        for start_index in range(0, len(frame) - window_size + 1, stride):
            end_index = start_index + window_size - 1
            window = frame.iloc[start_index : end_index + 1][list(feature_columns)].to_numpy(dtype=np.float32, copy=True)
            normalized_window = apply_normalizer(
                window,
                feature_columns=feature_columns,
                normalizer=normalizer,
            )
            inputs = torch.from_numpy(normalized_window).unsqueeze(0).to(device)
            reconstruction = model(inputs)["reconstruction"].detach().cpu().numpy()[0]
            residuals = np.square(normalized_window - reconstruction)
            group_scores = compute_group_scores(residuals, top_fraction=0.05)
            top_group = max(group_scores.items(), key=lambda item: item[1])[0]
            score_rows.append(
                {
                    "window_start_idx": start_index,
                    "window_end_idx": end_index,
                    "score": aggregate_top_timesteps(residuals, top_fraction=0.05),
                    "top_group": top_group,
                    **{f"{group_name}_score": value for group_name, value in group_scores.items()},
                }
            )
    return score_rows


def _summarize_clean_scores(
    score_rows: list[dict[str, object]],
    *,
    threshold: float,
    frame_length: int,
    required_consecutive: int,
    stride: int,
) -> dict[str, float]:
    positives = []
    alert_flags = []
    history: list[bool] = []
    for row in score_rows:
        positive = bool(float(row["score"]) > threshold)
        positives.append(positive)
        history.append(positive)
        alert_flags.append(passes_persistence_rule(history, required_consecutive=required_consecutive))

    alert_segments = _count_alert_segments(alert_flags)
    alert_durations = _alert_durations_seconds(alert_flags, stride=stride)
    duration_minutes = max(frame_length / 50.0 / 60.0, 1e-6)
    return {
        "threshold": threshold,
        "clean_alert_rate": float(np.mean(alert_flags)) if alert_flags else 0.0,
        "clean_alerts_per_min": float(alert_segments / duration_minutes),
        "clean_median_alert_duration_s": float(np.median(alert_durations)) if alert_durations else 0.0,
    }


def _evaluate_injected_event(
    *,
    score_rows: list[dict[str, object]],
    metadata: dict,
    threshold: float,
    required_consecutive: int,
    grace_period_s: float,
) -> dict[str, object]:
    history: list[bool] = []
    first_alert_row = None
    for row in score_rows:
        positive = bool(float(row["score"]) > threshold)
        history.append(positive)
        if passes_persistence_rule(history, required_consecutive=required_consecutive):
            alert_time_index = int(row["window_end_idx"])
            if alert_time_index >= metadata["start_index"]:
                first_alert_row = row
                break

    grace_steps = int(grace_period_s * 50.0)
    detected = False
    time_to_detect_s = float("inf")
    top_group_prediction = ""
    if first_alert_row is not None:
        alert_time_index = int(first_alert_row["window_end_idx"])
        detected = alert_time_index <= metadata["end_index"] + grace_steps
        time_to_detect_s = max(0.0, (alert_time_index - metadata["start_index"]) / 50.0)
        top_group_prediction = str(first_alert_row["top_group"])

    target_group = metadata["target_group"]
    if target_group in GROUP_SLICES and top_group_prediction:
        group_attribution_correct = float(top_group_prediction == target_group)
    else:
        group_attribution_correct = np.nan

    return {
        "anomaly_type": metadata["anomaly_type"],
        "severity": metadata["severity"],
        "target_group": target_group,
        "event_detected": float(detected),
        "time_to_detect_s": time_to_detect_s,
        "top_group_prediction": top_group_prediction,
        "group_attribution_correct": group_attribution_correct,
    }


def _summarize_injected_breakdown(anomaly_breakdown: pd.DataFrame) -> dict[str, float]:
    event_detection_rate = float(anomaly_breakdown["event_detected"].mean()) if not anomaly_breakdown.empty else 0.0
    detected_times = anomaly_breakdown.loc[anomaly_breakdown["event_detected"] > 0.0, "time_to_detect_s"]
    finite_detected_times = detected_times[np.isfinite(detected_times.to_numpy(dtype=float))]
    attribution = anomaly_breakdown["group_attribution_correct"].dropna()
    return {
        "event_detection_rate": event_detection_rate,
        "median_time_to_detect_s": float(np.median(finite_detected_times)) if len(finite_detected_times) else float("inf"),
        "p90_time_to_detect_s": float(np.percentile(finite_detected_times, 90)) if len(finite_detected_times) else float("inf"),
        "group_attribution_top1": float(attribution.mean()) if not attribution.empty else 0.0,
        "channel_top3_hit_rate": 0.0,
    }


def _count_alert_segments(alert_flags: list[bool]) -> int:
    segments = 0
    was_alerting = False
    for flag in alert_flags:
        if flag and not was_alerting:
            segments += 1
        was_alerting = flag
    return segments


def _alert_durations_seconds(alert_flags: list[bool], *, stride: int) -> list[float]:
    durations = []
    current_duration = 0
    for flag in alert_flags:
        if flag:
            current_duration += stride
        elif current_duration:
            durations.append(current_duration / 50.0)
            current_duration = 0
    if current_duration:
        durations.append(current_duration / 50.0)
    return durations


def _select_injection_start(*, frame_length: int, window_size: int, duration: int) -> int:
    lower_bound = min(frame_length - duration - 1, window_size + 10)
    upper_bound = frame_length - duration - 1
    if upper_bound < lower_bound:
        return max(0, upper_bound)
    return lower_bound


def _build_injection_context(frame: pd.DataFrame) -> dict[str, list[float]]:
    length = len(frame)
    return {
        "gt_speed_mps": frame["gt_speed_mps"].tolist() if "gt_speed_mps" in frame.columns else [1.0] * length,
        "gt_vertical_speed_mps": (
            frame["gt_vertical_speed_mps"].tolist() if "gt_vertical_speed_mps" in frame.columns else [0.1] * length
        ),
        "gt_yaw_rate_rps": frame["gt_yaw_rate_rps"].tolist() if "gt_yaw_rate_rps" in frame.columns else [0.2] * length,
    }
