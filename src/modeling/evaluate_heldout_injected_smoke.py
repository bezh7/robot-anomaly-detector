from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.modeling.anomaly_injection import (
    DURATIONS,
    MOTION_SPEED_THRESHOLD,
    VERTICAL_SPEED_THRESHOLD,
    YAW_RATE_THRESHOLD,
    build_injected_sequence,
)
from src.modeling.replay_dashboard_data import (
    _build_injection_context,
    _fit_full_dev_normalizer,
    _score_frame,
    load_final_model_bundle,
)
from src.modeling.modeling_dataset import load_feature_tables


ANOMALY_CATALOG: tuple[dict[str, str], ...] = (
    {
        "anomaly_type": "gyro_bias_step",
        "tier": "implemented",
        "target_group": "gyro",
        "injection_method": "Add a constant offset to all gyro axes for a short interval.",
    },
    {
        "anomaly_type": "gyro_bias_drift",
        "tier": "implemented",
        "target_group": "gyro",
        "injection_method": "Add a linearly increasing ramp to all gyro axes.",
    },
    {
        "anomaly_type": "accel_bias_drift",
        "tier": "implemented",
        "target_group": "accel",
        "injection_method": "Add a linearly increasing ramp to all accel axes.",
    },
    {
        "anomaly_type": "accel_freeze",
        "tier": "implemented",
        "target_group": "accel",
        "injection_method": "Hold all accel axes at the value from anomaly onset.",
    },
    {
        "anomaly_type": "gyro_freeze",
        "tier": "implemented",
        "target_group": "gyro",
        "injection_method": "Hold all gyro axes at the value from anomaly onset.",
    },
    {
        "anomaly_type": "noise_burst",
        "tier": "implemented",
        "target_group": "mixed",
        "injection_method": "Add Gaussian noise to both accel and gyro channels.",
    },
    {
        "anomaly_type": "clipping",
        "tier": "implemented-legacy",
        "target_group": "accel",
        "injection_method": "Legacy clipping with inverted severity semantics; preserved for backward compatibility.",
    },
    {
        "anomaly_type": "hard_clipping",
        "tier": "implemented",
        "target_group": "accel",
        "injection_method": "Clip accel channels to a bounded range with severity mapped to harsher lower clip bounds.",
    },
    {
        "anomaly_type": "packet_dropout",
        "tier": "implemented",
        "target_group": "mixed",
        "injection_method": "Replace live raw IMU samples with stale packet values over a short interval.",
    },
    {
        "anomaly_type": "sensor_lag",
        "tier": "implemented",
        "target_group": "accel",
        "injection_method": "Replace accel samples with lagged earlier accel values during active motion.",
    },
    {
        "anomaly_type": "timestamp_jitter",
        "tier": "implemented",
        "target_group": "mixed",
        "injection_method": "Warp raw IMU values with a non-uniform time axis to mimic jitter before resampling back.",
    },
    {
        "anomaly_type": "cross_axis_leakage",
        "tier": "implemented",
        "target_group": "accel",
        "injection_method": "Leak accel energy across axes to mimic transient cross-axis coupling.",
    },
    {
        "anomaly_type": "gyro_accel_inconsistency",
        "tier": "implemented",
        "target_group": "accel",
        "injection_method": "Rotate accel vectors during active motion while leaving gyro unchanged, creating cross-sensor inconsistency.",
    },
    {
        "anomaly_type": "impact_pulse",
        "tier": "implemented",
        "target_group": "accel",
        "injection_method": "Add a short Hanning pulse to vertical acceleration.",
    },
    {
        "anomaly_type": "vibration_burst",
        "tier": "implemented",
        "target_group": "accel",
        "injection_method": "Add a short sinusoidal burst to vertical acceleration.",
    },
    {
        "anomaly_type": "angular_rate_burst",
        "tier": "implemented",
        "target_group": "gyro",
        "injection_method": "Add a short sinusoidal burst to yaw rate.",
    },
    {
        "anomaly_type": "accel_bias_step",
        "tier": "planned",
        "target_group": "accel",
        "injection_method": "Add a constant offset to all accel axes for a fixed interval.",
    },
    {
        "anomaly_type": "accel_bias_drift",
        "tier": "planned",
        "target_group": "accel",
        "injection_method": "Add a slowly increasing ramp to accel axes.",
    },
    {
        "anomaly_type": "gyro_freeze",
        "tier": "planned",
        "target_group": "gyro",
        "injection_method": "Hold all gyro axes constant during motion.",
    },
    {
        "anomaly_type": "stuck_at_zero",
        "tier": "planned",
        "target_group": "mixed",
        "injection_method": "Force one or more channels to zero for a sustained interval.",
    },
    {
        "anomaly_type": "packet_dropout",
        "tier": "planned",
        "target_group": "mixed",
        "injection_method": "Replace a span with repeated previous values to mimic dropped packets.",
    },
    {
        "anomaly_type": "quantization_stair_step",
        "tier": "planned",
        "target_group": "mixed",
        "injection_method": "Reduce channel precision to coarse quantized bins over an interval.",
    },
    {
        "anomaly_type": "axis_sign_flip",
        "tier": "planned",
        "target_group": "gyro",
        "injection_method": "Invert one axis sign to mimic a sign convention or wiring fault.",
    },
    {
        "anomaly_type": "axis_swap",
        "tier": "planned",
        "target_group": "mixed",
        "injection_method": "Swap two physical axes to mimic miswired or misconfigured channels.",
    },
    {
        "anomaly_type": "timestamp_jitter",
        "tier": "planned",
        "target_group": "mixed",
        "injection_method": "Perturb effective sample timing and resample back to fixed rate.",
    },
    {
        "anomaly_type": "cross_axis_coupling",
        "tier": "planned",
        "target_group": "mixed",
        "injection_method": "Leak motion from one axis into another to mimic calibration error.",
    },
)

DEFAULT_ADAPTIVE_WINDOW_ROWS = 300
DEFAULT_ADAPTIVE_Z_THRESHOLD = 5.0
DEFAULT_ADAPTIVE_SCALE_FLOOR_RATIO = 0.05
DEFAULT_ADAPTIVE_MIN_SCALE = 1e-6


@dataclass(frozen=True)
class HeldoutInjectedSmokeResult:
    trial_metrics_path: Path
    summary_metrics_path: Path
    clean_summary_path: Path
    segment_candidates_path: Path
    anomaly_catalog_path: Path
    clean_timeline_path: Path
    trial_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    clean_summary: pd.DataFrame


def annotate_alerts(
    timeline: pd.DataFrame,
    *,
    threshold: float,
    required_consecutive: int,
    threshold_mode: str = "global",
    adaptive_window_rows: int = DEFAULT_ADAPTIVE_WINDOW_ROWS,
    adaptive_z_threshold: float = DEFAULT_ADAPTIVE_Z_THRESHOLD,
    adaptive_scale_floor_ratio: float = DEFAULT_ADAPTIVE_SCALE_FLOOR_RATIO,
    adaptive_min_scale: float = DEFAULT_ADAPTIVE_MIN_SCALE,
) -> pd.DataFrame:
    annotated = timeline.copy()

    if threshold_mode == "global":
        annotated["threshold"] = float(threshold)
        positives = annotated["score"].to_numpy(dtype=float) > float(threshold)
        annotated["adaptive_baseline_score"] = np.nan
        annotated["adaptive_scale"] = np.nan
        annotated["normalized_score"] = np.nan
        annotated["normalized_threshold"] = np.nan
    elif threshold_mode == "adaptive":
        annotated["threshold"] = np.nan
        annotated["normalized_threshold"] = float(adaptive_z_threshold)
        positives, baselines, scales, normalized_scores = _compute_adaptive_positives(
            annotated["score"].to_numpy(dtype=float),
            required_consecutive=required_consecutive,
            adaptive_window_rows=adaptive_window_rows,
            adaptive_z_threshold=adaptive_z_threshold,
            adaptive_scale_floor_ratio=adaptive_scale_floor_ratio,
            adaptive_min_scale=adaptive_min_scale,
        )
        annotated["adaptive_baseline_score"] = baselines
        annotated["adaptive_scale"] = scales
        annotated["normalized_score"] = normalized_scores
    else:
        raise ValueError(f"unsupported threshold_mode: {threshold_mode}")

    history: list[bool] = []
    alert_active: list[bool] = []
    for positive in positives.tolist():
        history.append(bool(positive))
        alert_active.append(len(history) >= required_consecutive and all(history[-required_consecutive:]))

    alert_onset = []
    was_active = False
    for active in alert_active:
        onset = bool(active and not was_active)
        alert_onset.append(onset)
        was_active = bool(active)

    annotated["positive"] = positives
    annotated["alert_active"] = alert_active
    annotated["alert_onset"] = alert_onset
    return annotated


def _compute_adaptive_positives(
    scores: np.ndarray,
    *,
    required_consecutive: int,
    adaptive_window_rows: int,
    adaptive_z_threshold: float,
    adaptive_scale_floor_ratio: float,
    adaptive_min_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    positives = np.zeros(scores.shape[0], dtype=bool)
    baselines = np.full(scores.shape[0], np.nan, dtype=float)
    scales = np.full(scores.shape[0], np.nan, dtype=float)
    normalized_scores = np.full(scores.shape[0], np.nan, dtype=float)

    history: list[bool] = []
    alert_active: list[bool] = []
    for index, score in enumerate(scores):
        previous_active = alert_active[-1] if alert_active else False
        if previous_active and index > 0:
            baseline = baselines[index - 1]
            scale = scales[index - 1]
        else:
            start_index = max(0, index - adaptive_window_rows)
            trailing_scores = scores[start_index:index]
            if trailing_scores.size == 0:
                baseline = score
                mad = 0.0
            else:
                baseline = float(np.median(trailing_scores))
                mad = float(np.median(np.abs(trailing_scores - baseline)))
            scale = max(
                1.4826 * mad,
                adaptive_min_scale,
                adaptive_scale_floor_ratio * max(abs(baseline), adaptive_min_scale),
            )

        normalized_score = (score - baseline) / scale if scale > 0.0 else 0.0
        positive = bool(normalized_score > adaptive_z_threshold and score > baseline)

        baselines[index] = baseline
        scales[index] = scale
        normalized_scores[index] = normalized_score
        positives[index] = positive

        history.append(positive)
        active = len(history) >= required_consecutive and all(history[-required_consecutive:])
        alert_active.append(active)

    return positives, baselines, scales, normalized_scores


def select_low_score_start_indices(
    *,
    clean_timeline: pd.DataFrame,
    frame_length: int,
    duration: int,
    candidate_count: int,
    stride: int,
    pre_context_rows: int = 150,
    post_context_rows: int = 150,
    min_gap_rows: int = 500,
    candidate_validator: Callable[[int], bool] | None = None,
) -> list[int]:
    candidates: list[dict[str, float | int]] = []
    latest_start = frame_length - duration - post_context_rows - 1
    for start_index in range(pre_context_rows, max(pre_context_rows, latest_start) + 1, stride):
        if candidate_validator is not None and not candidate_validator(start_index):
            continue

        region_start = start_index - pre_context_rows
        region_end = start_index + duration + post_context_rows
        region = clean_timeline.loc[
            (clean_timeline["window_end_idx"] >= region_start)
            & (clean_timeline["window_start_idx"] <= region_end)
        ]
        if region.empty or bool(region["alert_active"].any()):
            continue

        pre_region = clean_timeline.loc[
            (clean_timeline["window_end_idx"] >= region_start)
            & (clean_timeline["window_end_idx"] < start_index)
        ]
        baseline_score = float(pre_region["score"].median()) if not pre_region.empty else float(region["score"].median())
        candidates.append(
            {
                "start_index": int(start_index),
                "region_max_score": float(region["score"].max()),
                "baseline_score": baseline_score,
            }
        )

    ranked = sorted(candidates, key=lambda row: (row["region_max_score"], row["baseline_score"], row["start_index"]))
    selected: list[int] = []
    for row in ranked:
        start_index = int(row["start_index"])
        if any(abs(start_index - existing) < min_gap_rows for existing in selected):
            continue
        selected.append(start_index)
        if len(selected) >= candidate_count:
            break
    return selected


def evaluate_strict_injected_event(
    *,
    annotated_timeline: pd.DataFrame,
    metadata: dict[str, object],
    grace_period_s: float,
    pre_quiet_s: float = 1.0,
    sample_rate_hz: float = 50.0,
) -> dict[str, object]:
    start_index = int(metadata["start_index"])
    end_index = int(metadata["end_index"])
    grace_rows = int(grace_period_s * sample_rate_hz)
    pre_quiet_rows = int(pre_quiet_s * sample_rate_hz)

    pre_region = annotated_timeline.loc[
        (annotated_timeline["window_end_idx"] >= max(0, start_index - pre_quiet_rows))
        & (annotated_timeline["window_end_idx"] < start_index)
    ]
    preexisting_alert = bool(pre_region["alert_active"].any())
    pre_event_alert_onset_count = int(pre_region["alert_onset"].sum()) if "alert_onset" in pre_region else 0
    pre_event_alert_active_count = int(pre_region["alert_active"].sum()) if "alert_active" in pre_region else 0

    event_region = annotated_timeline.loc[
        (annotated_timeline["window_end_idx"] >= start_index)
        & (annotated_timeline["window_end_idx"] <= end_index + grace_rows)
    ]
    onset_region = event_region.loc[event_region["alert_onset"]]
    first_onset = onset_region.iloc[0] if not onset_region.empty else None

    detected = bool(first_onset is not None and not preexisting_alert)
    alert_trigger_group = str(first_onset["top_group"]) if first_onset is not None else ""
    alert_onset_idx = int(first_onset["window_end_idx"]) if first_onset is not None else -1
    time_to_detect_s = (
        max(0.0, (alert_onset_idx - start_index) / sample_rate_hz) if first_onset is not None and detected else float("inf")
    )

    during_region = annotated_timeline.loc[
        (annotated_timeline["window_end_idx"] >= start_index)
        & (annotated_timeline["window_start_idx"] <= end_index)
    ]
    post_region = annotated_timeline.loc[
        (annotated_timeline["window_start_idx"] > end_index)
        & (annotated_timeline["window_start_idx"] <= end_index + pre_quiet_rows)
    ]
    outside_event_region = annotated_timeline.loc[
        (annotated_timeline["window_end_idx"] < start_index)
        | (annotated_timeline["window_start_idx"] > end_index + grace_rows)
    ]
    outside_event_alert_onset_count = (
        int(outside_event_region["alert_onset"].sum()) if "alert_onset" in outside_event_region else 0
    )
    outside_event_alert_active_count = (
        int(outside_event_region["alert_active"].sum()) if "alert_active" in outside_event_region else 0
    )

    baseline_score = float(pre_region["score"].median()) if not pre_region.empty else float("nan")
    during_max_score = float(during_region["score"].max()) if not during_region.empty else float("nan")
    during_median_score = float(during_region["score"].median()) if not during_region.empty else float("nan")
    post_median_score = float(post_region["score"].median()) if not post_region.empty else float("nan")
    score_lift_abs = during_max_score - baseline_score if np.isfinite(during_max_score) and np.isfinite(baseline_score) else float("nan")
    score_lift_ratio = (
        during_max_score / baseline_score
        if np.isfinite(during_max_score) and np.isfinite(baseline_score) and abs(baseline_score) > 1e-12
        else float("nan")
    )

    target_group = str(metadata["target_group"])
    attribution_correct = (
        float(alert_trigger_group == target_group) if detected and target_group in {"quaternion", "gyro", "accel"} else np.nan
    )

    return {
        "anomaly_type": str(metadata["anomaly_type"]),
        "severity": str(metadata["severity"]),
        "target_group": target_group,
        "start_index": start_index,
        "end_index": end_index,
        "preexisting_alert": preexisting_alert,
        "pre_event_alert_onset_count": pre_event_alert_onset_count,
        "pre_event_alert_active_count": pre_event_alert_active_count,
        "outside_event_alert_onset_count": outside_event_alert_onset_count,
        "outside_event_alert_active_count": outside_event_alert_active_count,
        "event_detected_strict": float(detected),
        "time_to_detect_s": time_to_detect_s,
        "alert_trigger_group": alert_trigger_group,
        "group_attribution_correct": attribution_correct,
        "baseline_score": baseline_score,
        "during_max_score": during_max_score,
        "during_median_score": during_median_score,
        "post_median_score": post_median_score,
        "score_lift_abs": score_lift_abs,
        "score_lift_ratio": score_lift_ratio,
    }


def summarize_smoke_trials(trials: pd.DataFrame) -> pd.DataFrame:
    trials = trials.copy()
    for column_name in (
        "outside_event_alert_onset_count",
        "outside_event_alert_active_count",
    ):
        if column_name not in trials.columns:
            trials[column_name] = 0

    summary_rows: list[dict[str, object]] = []
    for (anomaly_type, severity), group in trials.groupby(["anomaly_type", "severity"], sort=True):
        detected_times = group.loc[group["event_detected_strict"] > 0.0, "time_to_detect_s"]
        finite_detected = detected_times[np.isfinite(detected_times.to_numpy(dtype=float))]
        attribution = group["group_attribution_correct"].dropna()
        summary_rows.append(
            {
                "anomaly_type": anomaly_type,
                "severity": severity,
                "trial_count": int(len(group)),
                "contaminated_trial_count": int(group["preexisting_alert"].sum()),
                "strict_detection_rate": float(group["event_detected_strict"].mean()),
                "median_time_to_detect_s": float(np.median(finite_detected)) if len(finite_detected) else float("inf"),
                "group_attribution_top1": float(attribution.mean()) if not attribution.empty else float("nan"),
                "median_outside_event_alert_onset_count": float(group["outside_event_alert_onset_count"].median()),
                "median_outside_event_alert_active_count": float(group["outside_event_alert_active_count"].median()),
                "trial_false_positive_onset_rate": float((group["outside_event_alert_onset_count"] > 0).mean()),
                "trial_false_positive_active_rate": float((group["outside_event_alert_active_count"] > 0).mean()),
                "median_score_lift_abs": float(group["score_lift_abs"].median()),
                "median_score_lift_ratio": float(group["score_lift_ratio"].median()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(["anomaly_type", "severity"]).reset_index(drop=True)


def summarize_clean_alerts(clean_timeline: pd.DataFrame) -> pd.DataFrame:
    row = {
        "window_count": int(len(clean_timeline)),
        "positive_count": int(clean_timeline["positive"].sum()),
        "alert_active_count": int(clean_timeline["alert_active"].sum()),
        "alert_onset_count": int(clean_timeline["alert_onset"].sum()),
        "positive_rate": float(clean_timeline["positive"].mean()) if len(clean_timeline) else 0.0,
        "alert_active_rate": float(clean_timeline["alert_active"].mean()) if len(clean_timeline) else 0.0,
        "threshold_mode": str(clean_timeline.attrs.get("threshold_mode", "")),
    }
    return pd.DataFrame([row])


def can_inject_in_context(
    *,
    context: dict[str, list[float]],
    anomaly_type: str,
    start_index: int,
    require_active_run: bool = True,
) -> bool:
    speed = float(context["gt_speed_mps"][start_index])
    vertical_speed = abs(float(context["gt_vertical_speed_mps"][start_index]))
    yaw_rate = abs(float(context["gt_yaw_rate_rps"][start_index]))
    active_run = (
        speed > MOTION_SPEED_THRESHOLD
        or vertical_speed > VERTICAL_SPEED_THRESHOLD
        or yaw_rate > YAW_RATE_THRESHOLD
    )
    if require_active_run and not active_run:
        return False

    if anomaly_type in {"impact_pulse", "vibration_burst"}:
        return speed > MOTION_SPEED_THRESHOLD or vertical_speed > VERTICAL_SPEED_THRESHOLD
    if anomaly_type in {"angular_rate_burst", "sensor_lag", "gyro_accel_inconsistency"}:
        return speed > MOTION_SPEED_THRESHOLD or yaw_rate > YAW_RATE_THRESHOLD
    return True


def run_heldout_injected_smoke(
    *,
    feature_root: Path | str,
    model_root: Path | str,
    output_root: Path | str,
    sequence_name: str = "final_challenge_ugv2",
    normalizer_feature_root: Path | str | None = None,
    anomaly_types: tuple[str, ...] | None = None,
    severities: tuple[str, ...] = ("small", "medium", "large"),
    placements_per_anomaly: int = 3,
    threshold_mode: str = "adaptive",
    threshold_percentile: float = 99.0,
    required_consecutive: int = 2,
    grace_period_s: float = 0.5,
    stride: int = 5,
    window_size: int = 150,
    local_context_rows: int = 300,
    adaptive_window_rows: int = DEFAULT_ADAPTIVE_WINDOW_ROWS,
    adaptive_z_threshold: float = DEFAULT_ADAPTIVE_Z_THRESHOLD,
    adaptive_scale_floor_ratio: float = DEFAULT_ADAPTIVE_SCALE_FLOOR_RATIO,
    device: str = "cpu",
) -> HeldoutInjectedSmokeResult:
    feature_root = Path(feature_root)
    normalizer_feature_root = Path(normalizer_feature_root) if normalizer_feature_root is not None else feature_root
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    bundle = load_final_model_bundle(model_root, device=device)
    feature_tables = load_feature_tables(feature_root)
    clean_frame = feature_tables[sequence_name]
    normalizer_tables = load_feature_tables(normalizer_feature_root)
    normalizer = _fit_full_dev_normalizer(
        feature_root=normalizer_feature_root,
        feature_tables=normalizer_tables,
        bundle=bundle,
    )
    clean_timeline = _score_frame(
        frame=clean_frame,
        feature_columns=bundle.feature_columns,
        normalizer=normalizer,
        model=bundle.model,
        stride=stride,
        window_size=window_size,
        device=device,
    )
    threshold = (
        float(np.percentile(clean_timeline["score"].to_numpy(dtype=float), threshold_percentile))
        if threshold_mode == "global"
        else float("nan")
    )
    annotated_clean_timeline = annotate_alerts(
        clean_timeline,
        threshold=threshold,
        required_consecutive=required_consecutive,
        threshold_mode=threshold_mode,
        adaptive_window_rows=adaptive_window_rows,
        adaptive_z_threshold=adaptive_z_threshold,
        adaptive_scale_floor_ratio=adaptive_scale_floor_ratio,
    )
    annotated_clean_timeline.attrs["threshold_mode"] = threshold_mode
    clean_timeline_path = output_root / "heldout_clean_timeline.csv"
    annotated_clean_timeline.to_csv(clean_timeline_path, index=False)
    clean_summary = summarize_clean_alerts(annotated_clean_timeline)
    clean_summary_path = output_root / "heldout_clean_summary.csv"
    clean_summary.to_csv(clean_summary_path, index=False)

    context = _build_injection_context(clean_frame)
    selected_anomaly_types = anomaly_types or tuple(
        entry["anomaly_type"] for entry in ANOMALY_CATALOG if entry["tier"] == "implemented"
    )

    candidate_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    for anomaly_type in selected_anomaly_types:
        duration = DURATIONS[anomaly_type]

        starts = select_low_score_start_indices(
            clean_timeline=annotated_clean_timeline,
            frame_length=len(clean_frame),
            duration=duration,
            candidate_count=placements_per_anomaly,
            stride=stride,
            candidate_validator=lambda start_index, anomaly_type=anomaly_type: can_inject_in_context(
                context=context,
                anomaly_type=anomaly_type,
                start_index=start_index,
            ),
        )
        for placement_index, start_index in enumerate(starts, start=1):
            candidate_rows.append(
                {
                    "anomaly_type": anomaly_type,
                    "placement_index": placement_index,
                    "start_index": start_index,
                    "duration": duration,
                }
            )
            for severity in severities:
                injected = build_injected_sequence(
                    clean_frame,
                    context=context,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    start_index=start_index,
                    seed=1000 + placement_index,
                )
                injected_timeline = _score_injected_segment(
                    frame=injected.frame,
                    feature_columns=bundle.feature_columns,
                    normalizer=normalizer,
                    model=bundle.model,
                    start_index=start_index,
                    end_index=int(injected.metadata["end_index"]),
                    stride=stride,
                    window_size=window_size,
                    local_context_rows=local_context_rows,
                    device=device,
                )
                annotated_injected = annotate_alerts(
                    injected_timeline,
                    threshold=threshold,
                    required_consecutive=required_consecutive,
                    threshold_mode=threshold_mode,
                    adaptive_window_rows=adaptive_window_rows,
                    adaptive_z_threshold=adaptive_z_threshold,
                    adaptive_scale_floor_ratio=adaptive_scale_floor_ratio,
                )
                trial_row = evaluate_strict_injected_event(
                    annotated_timeline=annotated_injected,
                    metadata=injected.metadata,
                    grace_period_s=grace_period_s,
                )
                trial_row["placement_index"] = placement_index
                trial_rows.append(trial_row)

    candidate_frame = pd.DataFrame(candidate_rows).sort_values(["anomaly_type", "placement_index"]).reset_index(drop=True)
    trial_frame = pd.DataFrame(trial_rows).sort_values(["anomaly_type", "severity", "placement_index"]).reset_index(drop=True)
    summary_frame = summarize_smoke_trials(trial_frame)

    segment_candidates_path = output_root / "heldout_smoke_segment_candidates.csv"
    trial_metrics_path = output_root / "heldout_smoke_trials.csv"
    summary_metrics_path = output_root / "heldout_smoke_summary.csv"
    anomaly_catalog_path = output_root / "imu_anomaly_catalog.json"

    candidate_frame.to_csv(segment_candidates_path, index=False)
    trial_frame.to_csv(trial_metrics_path, index=False)
    summary_frame.to_csv(summary_metrics_path, index=False)
    anomaly_catalog_path.write_text(json.dumps(list(ANOMALY_CATALOG), indent=2))

    return HeldoutInjectedSmokeResult(
        trial_metrics_path=trial_metrics_path,
        summary_metrics_path=summary_metrics_path,
        clean_summary_path=clean_summary_path,
        segment_candidates_path=segment_candidates_path,
        anomaly_catalog_path=anomaly_catalog_path,
        clean_timeline_path=clean_timeline_path,
        trial_metrics=trial_frame,
        summary_metrics=summary_frame,
        clean_summary=clean_summary,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict held-out injected-anomaly smoke test on the saved final model.")
    parser.add_argument("--feature-root", default="artifacts/replay_features_real")
    parser.add_argument("--model-root", default="artifacts/modeling_final")
    parser.add_argument("--output-root", default="artifacts/heldout_smoke")
    parser.add_argument("--sequence-name", default="final_challenge_ugv2")
    parser.add_argument("--normalizer-feature-root", default=None)
    parser.add_argument("--placements-per-anomaly", type=int, default=3)
    parser.add_argument("--threshold-mode", choices=("global", "adaptive"), default="adaptive")
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--required-consecutive", type=int, default=2)
    parser.add_argument("--grace-period-s", type=float, default=0.5)
    parser.add_argument("--adaptive-window-rows", type=int, default=DEFAULT_ADAPTIVE_WINDOW_ROWS)
    parser.add_argument("--adaptive-z-threshold", type=float, default=DEFAULT_ADAPTIVE_Z_THRESHOLD)
    parser.add_argument("--adaptive-scale-floor-ratio", type=float, default=DEFAULT_ADAPTIVE_SCALE_FLOOR_RATIO)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--anomaly-types", nargs="*", default=None)
    parser.add_argument("--local-context-rows", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_heldout_injected_smoke(
        feature_root=args.feature_root,
        model_root=args.model_root,
        output_root=args.output_root,
        sequence_name=args.sequence_name,
        normalizer_feature_root=args.normalizer_feature_root,
        anomaly_types=tuple(args.anomaly_types) if args.anomaly_types else None,
        placements_per_anomaly=args.placements_per_anomaly,
        threshold_mode=args.threshold_mode,
        threshold_percentile=args.threshold_percentile,
        required_consecutive=args.required_consecutive,
        grace_period_s=args.grace_period_s,
        local_context_rows=args.local_context_rows,
        adaptive_window_rows=args.adaptive_window_rows,
        adaptive_z_threshold=args.adaptive_z_threshold,
        adaptive_scale_floor_ratio=args.adaptive_scale_floor_ratio,
        device=args.device,
    )
    print(result.summary_metrics.to_csv(index=False))


def _score_injected_segment(
    *,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    normalizer: dict[str, object],
    model,
    start_index: int,
    end_index: int,
    stride: int,
    window_size: int,
    local_context_rows: int,
    device: str,
) -> pd.DataFrame:
    segment_start = max(0, start_index - (window_size + local_context_rows))
    segment_end = min(len(frame) - 1, end_index + window_size + local_context_rows)
    segment_frame = frame.iloc[segment_start : segment_end + 1].reset_index(drop=True)
    segment_timeline = _score_frame(
        frame=segment_frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        model=model,
        stride=stride,
        window_size=window_size,
        device=device,
    )
    segment_timeline["window_start_idx"] = segment_timeline["window_start_idx"] + segment_start
    segment_timeline["window_end_idx"] = segment_timeline["window_end_idx"] + segment_start
    return segment_timeline


if __name__ == "__main__":
    main()
