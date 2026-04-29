from pathlib import Path

import pandas as pd

from src.modeling.evaluate_heldout_injected_smoke import (
    annotate_alerts,
    can_inject_in_context,
    evaluate_strict_injected_event,
    select_low_score_start_indices,
    summarize_clean_alerts,
    summarize_smoke_trials,
)


def test_select_low_score_start_indices_prefers_quiet_non_alert_regions():
    clean_timeline = pd.DataFrame(
        {
            "window_start_idx": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "window_end_idx": [9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
            "score": [10.0, 9.0, 1.0, 1.2, 1.1, 1.0, 8.0, 7.0, 0.8, 0.9],
            "alert_active": [True, True, False, False, False, False, True, True, False, False],
        }
    )

    starts = select_low_score_start_indices(
        clean_timeline=clean_timeline,
        frame_length=140,
        duration=10,
        candidate_count=2,
        stride=10,
        pre_context_rows=10,
        post_context_rows=10,
        min_gap_rows=40,
    )

    assert starts == [90, 30]


def test_evaluate_strict_injected_event_requires_new_alert_onset():
    timeline = pd.DataFrame(
        {
            "window_start_idx": [0, 5, 10, 15, 20, 25],
            "window_end_idx": [9, 14, 19, 24, 29, 34],
            "score": [0.1, 0.2, 0.3, 0.9, 1.1, 1.2],
            "alert_active": [False, False, False, False, True, True],
            "alert_onset": [False, False, False, False, True, False],
            "top_group": ["gyro", "gyro", "gyro", "gyro", "accel", "accel"],
        }
    )
    result = evaluate_strict_injected_event(
        annotated_timeline=timeline,
        metadata={"anomaly_type": "accel_freeze", "severity": "medium", "start_index": 20, "end_index": 28, "target_group": "accel"},
        grace_period_s=0.5,
        pre_quiet_s=0.2,
        sample_rate_hz=50.0,
    )

    assert result["preexisting_alert"] is False
    assert result["pre_event_alert_onset_count"] == 0
    assert result["outside_event_alert_onset_count"] == 0
    assert result["event_detected_strict"] == 1.0
    assert result["alert_trigger_group"] == "accel"
    assert result["group_attribution_correct"] == 1.0
    assert result["time_to_detect_s"] == 0.18


def test_annotate_alerts_adaptive_threshold_captures_local_surprise():
    timeline = pd.DataFrame(
        {
            "window_start_idx": [0, 5, 10, 15, 20, 25, 30],
            "window_end_idx": [4, 9, 14, 19, 24, 29, 34],
            "score": [40.0, 42.0, 1.0, 1.1, 1.0, 5.0, 6.0],
        }
    )

    annotated = annotate_alerts(
        timeline,
        threshold=float("nan"),
        required_consecutive=2,
        threshold_mode="adaptive",
        adaptive_window_rows=3,
        adaptive_z_threshold=5.0,
        adaptive_scale_floor_ratio=0.05,
    )

    assert bool(annotated.loc[5, "positive"]) is True
    assert bool(annotated.loc[6, "positive"]) is True
    assert bool(annotated.loc[6, "alert_onset"]) is True


def test_annotate_alerts_adaptive_threshold_ignores_tiny_local_lift():
    timeline = pd.DataFrame(
        {
            "window_start_idx": [0, 5, 10, 15, 20],
            "window_end_idx": [4, 9, 14, 19, 24],
            "score": [1.0, 1.0, 1.0, 1.001, 1.0012],
        }
    )

    annotated = annotate_alerts(
        timeline,
        threshold=float("nan"),
        required_consecutive=2,
        threshold_mode="adaptive",
        adaptive_window_rows=3,
        adaptive_z_threshold=5.0,
        adaptive_scale_floor_ratio=0.05,
    )

    assert bool(annotated["positive"].any()) is False
    assert bool(annotated["alert_active"].any()) is False


def test_summarize_smoke_trials_groups_by_anomaly_and_severity():
    trials = pd.DataFrame(
        [
            {"anomaly_type": "clipping", "severity": "small", "event_detected_strict": 1.0, "time_to_detect_s": 0.2, "group_attribution_correct": 1.0, "preexisting_alert": False, "score_lift_abs": 2.0, "score_lift_ratio": 2.5},
            {"anomaly_type": "clipping", "severity": "small", "event_detected_strict": 0.0, "time_to_detect_s": float("inf"), "group_attribution_correct": float("nan"), "preexisting_alert": False, "score_lift_abs": 1.0, "score_lift_ratio": 1.2},
        ]
    )

    summary = summarize_smoke_trials(trials)

    assert list(summary["anomaly_type"]) == ["clipping"]
    assert float(summary.loc[0, "strict_detection_rate"]) == 0.5
    assert int(summary.loc[0, "trial_count"]) == 2


def test_summarize_clean_alerts_reports_false_positive_burden():
    clean_timeline = pd.DataFrame(
        {
            "positive": [False, True, True, False],
            "alert_active": [False, False, True, False],
            "alert_onset": [False, False, True, False],
        }
    )
    clean_timeline.attrs["threshold_mode"] = "adaptive"

    summary = summarize_clean_alerts(clean_timeline)

    assert int(summary.loc[0, "window_count"]) == 4
    assert int(summary.loc[0, "positive_count"]) == 2
    assert int(summary.loc[0, "alert_active_count"]) == 1
    assert int(summary.loc[0, "alert_onset_count"]) == 1
    assert float(summary.loc[0, "alert_active_rate"]) == 0.25
    assert summary.loc[0, "threshold_mode"] == "adaptive"


def test_can_inject_in_context_requires_active_run():
    context = {
        "gt_speed_mps": [0.0, 0.5],
        "gt_vertical_speed_mps": [0.0, 0.0],
        "gt_yaw_rate_rps": [0.0, 0.2],
    }

    assert can_inject_in_context(context=context, anomaly_type="accel_freeze", start_index=0) is False
    assert can_inject_in_context(context=context, anomaly_type="accel_freeze", start_index=1) is True
