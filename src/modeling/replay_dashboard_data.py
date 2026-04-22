from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.feature_contract import DEFAULT_INFERENCE_STRIDE, DEFAULT_WINDOW_SIZE, RAW_IMU_FEATURES
from src.features.fit_normalizers import fit_normalizer
from src.modeling.anomaly_injection import DURATIONS, build_injected_sequence
from src.modeling.model_scoring import (
    aggregate_top_timesteps,
    compute_group_scores,
    fit_percentile_threshold,
    passes_persistence_rule,
)
from src.modeling.modeling_dataset import apply_normalizer, load_feature_tables
from src.modeling.train_anomaly_model import load_model_from_checkpoint


@dataclass(frozen=True)
class FinalModelBundle:
    experiment_id: str
    run_dir: Path
    checkpoint_path: Path
    config_path: Path
    config: dict[str, object]
    feature_columns: tuple[str, ...]
    model: torch.nn.Module


@dataclass(frozen=True)
class ReplayTimelineResult:
    sequence_name: str
    replay_frame: pd.DataFrame
    timeline: pd.DataFrame
    trace_columns: tuple[str, ...]
    threshold: float
    anomaly_events: tuple[dict[str, object], ...]


DEMO_ANOMALY_PLAN: tuple[tuple[str, str], ...] = (
    ("gyro_bias_drift", "medium"),
    ("accel_freeze", "medium"),
    ("clipping", "medium"),
)


def load_final_model_bundle(model_root: Path | str, *, device: str = "cpu") -> FinalModelBundle:
    model_root = Path(model_root)
    config_paths = sorted(model_root.glob("*/full_dev/config.json"))
    if not config_paths:
        raise FileNotFoundError(f"no final model config found under {model_root}")
    if len(config_paths) > 1:
        raise ValueError(f"expected exactly one final model under {model_root}, found {len(config_paths)}")

    config_path = config_paths[0]
    run_dir = config_path.parent
    checkpoint_path = run_dir / "final_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing final checkpoint: {checkpoint_path}")

    config = json.loads(config_path.read_text())
    model, _checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    return FinalModelBundle(
        experiment_id=str(config["experiment_id"]),
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        config=config,
        feature_columns=tuple(config["feature_columns"]),
        model=model,
    )


def list_replay_sequences(feature_root: Path | str) -> list[str]:
    feature_tables = load_feature_tables(feature_root)
    return sorted(feature_tables.keys())


def build_replay_timeline(
    *,
    feature_root: Path | str,
    model_root: Path | str,
    sequence_name: str,
    normalizer_feature_root: Path | str | None = None,
    replay_mode: str = "clean",
    seed: int = 7,
    threshold_percentile: float = 99.0,
    required_consecutive: int = 2,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_INFERENCE_STRIDE,
    device: str = "cpu",
) -> ReplayTimelineResult:
    feature_root = Path(feature_root)
    normalizer_feature_root = Path(normalizer_feature_root) if normalizer_feature_root is not None else feature_root
    bundle = load_final_model_bundle(model_root, device=device)
    feature_tables = load_feature_tables(feature_root)
    if sequence_name not in feature_tables:
        raise ValueError(f"unknown sequence: {sequence_name}")

    clean_frame = feature_tables[sequence_name]
    if len(clean_frame) < window_size:
        raise ValueError(f"sequence {sequence_name} shorter than window size {window_size}")

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
    threshold = fit_percentile_threshold(clean_timeline["score"].tolist(), threshold_percentile)

    anomaly_events: tuple[dict[str, object], ...] = ()
    replay_frame = clean_frame
    if replay_mode == "clean":
        replay_timeline = clean_timeline.copy()
    elif replay_mode == "demo":
        replay_frame, anomaly_events = _build_demo_replay_frame(
            frame=clean_frame,
            seed=seed,
            window_size=window_size,
        )
        replay_timeline = _score_frame(
            frame=replay_frame,
            feature_columns=bundle.feature_columns,
            normalizer=normalizer,
            model=bundle.model,
            stride=stride,
            window_size=window_size,
            device=device,
        )
    else:
        raise ValueError(f"unsupported replay mode: {replay_mode}")

    replay_timeline = _annotate_timeline(
        replay_timeline,
        threshold=threshold,
        required_consecutive=required_consecutive,
        anomaly_events=anomaly_events,
    )
    trace_columns = tuple(column for column in RAW_IMU_FEATURES if column in replay_frame.columns)
    return ReplayTimelineResult(
        sequence_name=sequence_name,
        replay_frame=replay_frame,
        timeline=replay_timeline,
        trace_columns=trace_columns,
        threshold=threshold,
        anomaly_events=anomaly_events,
    )


def _fit_full_dev_normalizer(
    *,
    feature_root: Path,
    feature_tables: dict[str, pd.DataFrame],
    bundle: FinalModelBundle,
) -> dict[str, object]:
    split_manifest_path = feature_root / "split_manifest.json"
    training_sequences: list[str]
    if split_manifest_path.exists():
        split_manifest = json.loads(split_manifest_path.read_text())
        training_sequences = sorted(
            {
                str(sequence_name)
                for fold in split_manifest.get("folds", [])
                for sequence_name in fold.get("training_sequences", []) + [fold.get("validation_sequence")]
                if sequence_name in feature_tables
            }
        )
    else:
        training_sequences = sorted(feature_tables.keys())

    return fit_normalizer(
        feature_tables=feature_tables,
        training_sequences=training_sequences,
        feature_columns=list(bundle.feature_columns),
        mode=str(bundle.config["normalization"]),
    )


def _score_frame(
    *,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    normalizer: dict[str, object],
    model: torch.nn.Module,
    stride: int,
    window_size: int,
    device: str,
    batch_size: int = 256,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if len(frame) < window_size:
        return pd.DataFrame(rows)

    feature_matrix = frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float32, copy=True)
    normalized_matrix = apply_normalizer(feature_matrix, feature_columns=feature_columns, normalizer=normalizer)
    start_indices = list(range(0, len(frame) - window_size + 1, stride))
    model.eval()
    with torch.no_grad():
        first_timestamp_ns = int(frame["timestamp_ns"].iloc[0]) if "timestamp_ns" in frame.columns else 0
        for batch_start in range(0, len(start_indices), batch_size):
            batch_indices = start_indices[batch_start : batch_start + batch_size]
            batch_windows = np.stack(
                [normalized_matrix[start_index : start_index + window_size] for start_index in batch_indices],
                axis=0,
            )
            inputs = torch.from_numpy(batch_windows).to(device)
            reconstructions = model(inputs)["reconstruction"].detach().cpu().numpy()
            residual_batch = np.square(batch_windows - reconstructions)

            for row_index, start_index in enumerate(batch_indices):
                end_index = start_index + window_size - 1
                residuals = residual_batch[row_index]
                group_scores = compute_group_scores(residuals, top_fraction=0.05)
                rows.append(
                    {
                        "window_start_idx": start_index,
                        "window_end_idx": end_index,
                        "timestamp_ns": int(frame["timestamp_ns"].iloc[end_index]) if "timestamp_ns" in frame.columns else end_index,
                        "time_s": (
                            int(frame["timestamp_ns"].iloc[end_index]) - first_timestamp_ns
                        ) / 1e9
                        if "timestamp_ns" in frame.columns
                        else end_index / 50.0,
                        "score": aggregate_top_timesteps(residuals, top_fraction=0.05),
                        "top_group": max(group_scores.items(), key=lambda item: item[1])[0],
                        **{f"{group_name}_score": value for group_name, value in group_scores.items()},
                    }
                )
    return pd.DataFrame(rows)


def _annotate_timeline(
    timeline: pd.DataFrame,
    *,
    threshold: float,
    required_consecutive: int,
    anomaly_events: tuple[dict[str, object], ...],
) -> pd.DataFrame:
    annotated = timeline.copy()
    annotated["threshold"] = threshold

    history: list[bool] = []
    alert_flags = []
    for score in annotated["score"].tolist():
        history.append(float(score) > threshold)
        alert_flags.append(passes_persistence_rule(history, required_consecutive=required_consecutive))
    annotated["alert_active"] = alert_flags

    if not anomaly_events:
        annotated["anomaly_active"] = False
        annotated["target_group"] = ""
        annotated["anomaly_type"] = ""
    else:
        annotated["anomaly_active"] = False
        annotated["target_group"] = ""
        annotated["anomaly_type"] = ""
        for event in anomaly_events:
            start_index = int(event["start_index"])
            end_index = int(event["end_index"])
            active = (annotated["window_end_idx"] >= start_index) & (annotated["window_start_idx"] <= end_index)
            annotated.loc[active, "anomaly_active"] = True
            annotated.loc[active, "target_group"] = str(event["target_group"])
            annotated.loc[active, "anomaly_type"] = str(event["anomaly_type"])
    return annotated


def _build_demo_replay_frame(
    *,
    frame: pd.DataFrame,
    seed: int,
    window_size: int,
) -> tuple[pd.DataFrame, tuple[dict[str, object], ...]]:
    replay_frame = frame.copy(deep=True)
    context = _build_injection_context(frame)
    events: list[dict[str, object]] = []
    schedule = _build_demo_injection_schedule(frame_length=len(frame), window_size=window_size)
    for event_index, (anomaly_type, severity, start_index) in enumerate(schedule):
        injected = build_injected_sequence(
            replay_frame,
            context=context,
            anomaly_type=anomaly_type,
            severity=severity,
            start_index=start_index,
            seed=seed + event_index,
        )
        replay_frame = injected.frame
        events.append(injected.metadata)
    return replay_frame, tuple(events)


def _build_demo_injection_schedule(*, frame_length: int, window_size: int) -> tuple[tuple[str, str, int], ...]:
    available_end = frame_length - 1
    if available_end <= window_size + 100:
        raise ValueError("held-out replay sequence is too short for the demo anomaly schedule")

    usable_start = window_size + 50
    usable_end = available_end - 50
    anchor_points = np.linspace(usable_start, usable_end, num=len(DEMO_ANOMALY_PLAN) + 2, dtype=int)[1:-1]

    schedule: list[tuple[str, str, int]] = []
    for anchor, (anomaly_type, severity) in zip(anchor_points, DEMO_ANOMALY_PLAN, strict=True):
        duration = DURATIONS[anomaly_type]
        start_index = min(max(usable_start, int(anchor)), max(usable_start, usable_end - duration))
        schedule.append((anomaly_type, severity, start_index))
    return tuple(schedule)


def _build_injection_context(frame: pd.DataFrame) -> dict[str, list[float]]:
    length = len(frame)
    return {
        "gt_speed_mps": frame["gt_speed_mps"].tolist() if "gt_speed_mps" in frame.columns else [1.0] * length,
        "gt_vertical_speed_mps": (
            frame["gt_vertical_speed_mps"].tolist() if "gt_vertical_speed_mps" in frame.columns else [0.1] * length
        ),
        "gt_yaw_rate_rps": frame["gt_yaw_rate_rps"].tolist() if "gt_yaw_rate_rps" in frame.columns else [0.2] * length,
    }
