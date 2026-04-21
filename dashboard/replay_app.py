from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.modeling.replay_dashboard_data import build_replay_timeline, list_replay_sequences, load_final_model_bundle

DEFAULT_FEATURE_ROOT = Path("artifacts/replay_features_real")
DEFAULT_MODEL_ROOT = Path("artifacts/modeling_final")
PLOT_TRACE_COLUMNS = ("ang_vel_x", "ang_vel_y", "ang_vel_z", "lin_acc_x", "lin_acc_y", "lin_acc_z")
PLAYBACK_INTERVAL_S = 0.1
WINDOW_RATE_HZ = 50.0
PLAYBACK_SPEEDS = {
    "1x": 1.0,
    "4x": 4.0,
    "10x": 10.0,
}


def main() -> None:
    st.set_page_config(page_title="Replay Dashboard", layout="wide")
    st.title("Replay Dashboard")

    feature_root = DEFAULT_FEATURE_ROOT
    model_root = DEFAULT_MODEL_ROOT

    try:
        bundle = load_final_model_bundle(model_root)
        sequence_names = list_replay_sequences(feature_root)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if not sequence_names:
        st.error(f"no replay sequences found under {feature_root}")
        st.stop()

    if "final_challenge_ugv2" in sequence_names:
        sequence_name = "final_challenge_ugv2"
        st.caption(f"Replay source: `{sequence_name}` from `{feature_root}`")
    else:
        sequence_name = st.selectbox("Sequence", sequence_names, index=0)
    replay_mode = st.radio("Replay mode", ("clean", "demo"), horizontal=True)

    selection_key = f"{bundle.experiment_id}|{sequence_name}|{replay_mode}"
    if st.session_state.get("selection_key") != selection_key:
        st.session_state.selection_key = selection_key
        st.session_state.playhead = 0
        st.session_state.playing = False
        st.session_state.play_last_tick_at = None
        st.session_state.speed_label = "1x"
        st.session_state.scrub_time = None
        st.session_state.scrub_time_playhead = None

    try:
        replay = _load_replay(
            feature_root=str(feature_root),
            model_root=str(model_root),
            sequence_name=sequence_name,
            replay_mode=replay_mode,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    max_index = max(0, len(replay.timeline) - 1)
    control_columns = st.columns([1, 1, 1, 2])
    if control_columns[0].button("Play", width="stretch"):
        st.session_state.playing = True
        st.session_state.play_last_tick_at = time.perf_counter()
    if control_columns[1].button("Pause", width="stretch"):
        st.session_state.playing = False
        st.session_state.play_last_tick_at = None
    if control_columns[2].button("Reset", width="stretch"):
        st.session_state.playing = False
        st.session_state.play_last_tick_at = None
        st.session_state.playhead = 0
        st.rerun()
    speed_label = control_columns[3].radio(
        "Speed",
        options=tuple(PLAYBACK_SPEEDS.keys()),
        horizontal=True,
        key="speed_label",
        label_visibility="collapsed",
    )
    playhead = min(int(st.session_state.get("playhead", 0)), max_index)
    st.session_state.playhead = playhead
    _sync_scrub_state_before_render(replay, playhead=playhead)
    next_playhead = _render_dashboard(replay, bundle=bundle, playhead=playhead)
    if next_playhead is not None and next_playhead != playhead:
        st.session_state.playing = False
        st.session_state.play_last_tick_at = None
        st.session_state.playhead = next_playhead
        st.rerun()

    if bool(st.session_state.get("playing", False)):
        if playhead >= max_index:
            st.session_state.playing = False
            st.session_state.play_last_tick_at = None
        else:
            interval_s = PLAYBACK_INTERVAL_S / PLAYBACK_SPEEDS[speed_label]
            now = time.perf_counter()
            last_tick_at = float(st.session_state.get("play_last_tick_at") or now)
            elapsed_s = max(0.0, now - last_tick_at)
            if elapsed_s < interval_s:
                time.sleep(interval_s - elapsed_s)
                now = time.perf_counter()
                elapsed_s = max(0.0, now - last_tick_at)
            step_count = max(1, int(elapsed_s / interval_s))
            next_playhead = min(max_index, playhead + step_count)
            st.session_state.playhead = next_playhead
            st.session_state.play_last_tick_at = now
            if next_playhead >= max_index:
                st.session_state.playing = False
                st.session_state.play_last_tick_at = None
            st.rerun()


@st.cache_data(show_spinner=False)
def _load_replay(*, feature_root: str, model_root: str, sequence_name: str, replay_mode: str):
    return build_replay_timeline(
        feature_root=feature_root,
        model_root=model_root,
        sequence_name=sequence_name,
        replay_mode=replay_mode,
    )


def _render_dashboard(replay, *, bundle, playhead: int) -> int | None:
    current_row = replay.timeline.iloc[playhead]
    total_duration_s = float(replay.replay_frame["timestamp_ns"].iloc[-1] - replay.replay_frame["timestamp_ns"].iloc[0]) / 1e9
    progress_pct = min(100.0, 100.0 * float(current_row["time_s"]) / max(total_duration_s, 1e-6))
    current_alert = _current_alert_segment(replay.timeline, playhead=playhead)

    if current_alert is not None:
        banner = (
            f"ALERT ON — trigger group: {current_alert['trigger_group']} | "
            f"current top group: {current_row['top_group']}"
        )
        if current_alert["target_group"]:
            banner += f" | target group: {current_alert['target_group']}"
        st.error(banner)
    else:
        st.success("No active alert")

    metrics = st.columns(7)
    metrics[0].metric("Sequence", replay.sequence_name)
    metrics[1].metric("Time", f"{float(current_row['time_s']):.1f}s")
    metrics[2].metric("Window", f"{int(current_row['window_start_idx'])/WINDOW_RATE_HZ:.1f}-{int(current_row['window_end_idx'])/WINDOW_RATE_HZ:.1f}s")
    metrics[3].metric("Progress", f"{progress_pct:.1f}%")
    metrics[4].metric("Threshold", f"{replay.threshold:.4f}")
    metrics[5].metric("Score", f"{float(current_row['score']):.4f}")
    metrics[6].metric("Alert", "on" if bool(current_row["alert_active"]) else "off")

    status = st.columns(5)
    status[0].metric("Current top group", str(current_row["top_group"]))
    status[1].metric("Trigger group", current_alert["trigger_group"] if current_alert is not None else "none")
    status[2].metric("Target group", str(current_row.get("target_group", "")) or "none")
    status[3].metric("Anomaly", str(current_row.get("anomaly_type", "")) or "none")
    status[4].metric("Run length", f"{total_duration_s:.1f}s")

    st.plotly_chart(_build_score_figure(replay, playhead=playhead), width="stretch")
    selected_time = st.slider(
        "Sequence time",
        min_value=float(replay.timeline["time_s"].iloc[0]),
        max_value=float(replay.timeline["time_s"].iloc[-1]),
        key="scrub_time",
        step=PLAYBACK_INTERVAL_S,
        format="%.1f s",
    )
    if abs(selected_time - float(current_row["time_s"])) > (PLAYBACK_INTERVAL_S / 2):
        return _nearest_playhead(replay.timeline["time_s"], selected_time)

    figure_columns = st.columns(2)
    figure_columns[0].plotly_chart(_build_trace_figure(replay, playhead=playhead), width="stretch")
    figure_columns[1].plotly_chart(_build_group_score_figure(replay, playhead=playhead), width="stretch")
    return None


def _build_score_figure(replay, *, playhead: int) -> go.Figure:
    timeline = replay.timeline
    current_time = float(timeline.iloc[playhead]["time_s"])
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=timeline["time_s"], y=timeline["score"], mode="lines", name="score"))
    figure.add_trace(
        go.Scatter(
            x=timeline["time_s"],
            y=timeline["threshold"],
            mode="lines",
            name="threshold",
            line={"dash": "dash"},
        )
    )
    figure.add_vline(x=current_time, line_width=3, line_color="red")
    figure.add_annotation(
        x=current_time,
        y=1.06,
        yref="paper",
        text=f"t={current_time:.1f}s",
        showarrow=False,
        font={"size": 12},
        bgcolor="rgba(255,240,240,0.92)",
        bordercolor="red",
        borderwidth=1,
    )
    for event in replay.anomaly_events:
        start_time = event["start_index"] / WINDOW_RATE_HZ
        end_time = event["end_index"] / WINDOW_RATE_HZ
        figure.add_vrect(x0=start_time, x1=end_time, fillcolor="rgba(255,0,0,0.08)", line_width=0)
        figure.add_annotation(
            x=(start_time + end_time) / 2.0,
            y=1.03,
            yref="paper",
            text=f"injected: {event['anomaly_type']} ({event['target_group']})",
            showarrow=False,
            font={"size": 12, "color": "#7a0000"},
            bgcolor="rgba(255,235,235,0.92)",
            bordercolor="rgba(180,0,0,0.75)",
            borderwidth=1,
        )
    for segment in _alert_segments(timeline):
        figure.add_vrect(
            x0=segment["start_time_s"],
            x1=segment["end_time_s"],
            fillcolor="rgba(255,165,0,0.10)",
            line_width=0,
        )
        figure.add_annotation(
            x=segment["start_time_s"],
            y=float(segment["peak_score"]),
            text=f"alert: {segment['trigger_group']}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-30,
        )
    figure.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="time (s)",
        yaxis_title="anomaly score",
        showlegend=True,
    )
    return figure


def _build_trace_figure(replay, *, playhead: int) -> go.Figure:
    frame = replay.replay_frame
    timeline_row = replay.timeline.iloc[playhead]
    start_index = max(0, int(timeline_row["window_start_idx"]))
    end_index = min(len(frame) - 1, int(timeline_row["window_end_idx"]))
    window_frame = frame.iloc[start_index : end_index + 1]
    time_s = (window_frame["timestamp_ns"] - int(frame["timestamp_ns"].iloc[0])) / 1e9
    trace_columns = [column for column in PLOT_TRACE_COLUMNS if column in frame.columns]

    figure = go.Figure()
    for column in trace_columns:
        figure.add_trace(go.Scatter(x=time_s, y=window_frame[column], mode="lines", name=column))
    figure.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="time (s)",
        yaxis_title="current 3 s window",
        showlegend=True,
    )
    return figure


def _build_group_score_figure(replay, *, playhead: int) -> go.Figure:
    current_row = replay.timeline.iloc[playhead]
    figure = go.Figure()
    labels = ["quaternion", "gyro", "accel"]
    values = [float(current_row["quaternion_score"]), float(current_row["gyro_score"]), float(current_row["accel_score"])]
    top_group = str(current_row["top_group"])
    color_map = {
        "quaternion": "#bdbdbd",
        "gyro": "#1f77b4",
        "accel": "#ff7f0e",
    }
    colors = [color_map[label] if label == top_group else "#d9d9d9" for label in labels]
    figure.add_trace(go.Bar(x=labels, y=values, marker_color=colors))
    figure.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_title="group score",
        showlegend=False,
    )
    return figure


def _alert_segments(timeline) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    start_index: int | None = None
    for row_index, is_alert in enumerate(timeline["alert_active"].tolist()):
        if is_alert and start_index is None:
            start_index = row_index
        elif not is_alert and start_index is not None:
            segments.append(_build_alert_segment(timeline, start_index=start_index, end_index=row_index - 1))
            start_index = None
    if start_index is not None:
        segments.append(_build_alert_segment(timeline, start_index=start_index, end_index=len(timeline) - 1))
    return segments


def _build_alert_segment(timeline, *, start_index: int, end_index: int) -> dict[str, object]:
    segment = timeline.iloc[start_index : end_index + 1]
    trigger_row = timeline.iloc[start_index]
    peak_row = segment.loc[segment["score"].idxmax()]
    target_group = str(segment["target_group"].replace("", pd.NA).dropna().iloc[0]) if "target_group" in segment and segment["target_group"].replace("", pd.NA).notna().any() else ""
    return {
        "start_index": start_index,
        "end_index": end_index,
        "start_time_s": float(trigger_row["time_s"]),
        "end_time_s": float(segment.iloc[-1]["time_s"]),
        "trigger_group": str(trigger_row["top_group"]),
        "target_group": target_group,
        "peak_score": float(peak_row["score"]),
    }


def _current_alert_segment(timeline, *, playhead: int) -> dict[str, object] | None:
    for segment in _alert_segments(timeline):
        if segment["start_index"] <= playhead <= segment["end_index"]:
            return segment
    return None


def _nearest_playhead(time_series, target_time_s: float) -> int:
    values = time_series.to_numpy(dtype=float)
    return int(np.abs(values - float(target_time_s)).argmin())


def _sync_scrub_state_before_render(replay, *, playhead: int) -> None:
    if st.session_state.get("scrub_time_playhead") == playhead and st.session_state.get("scrub_time") is not None:
        return
    current_time = float(replay.timeline.iloc[playhead]["time_s"])
    st.session_state.scrub_time = current_time
    st.session_state.scrub_time_playhead = playhead


if __name__ == "__main__":
    main()
