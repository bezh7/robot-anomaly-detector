from io import StringIO

from src.profiling import infer_platform_hint, profile_imu_csv, summarize_timestamps


def test_infer_platform_hint_from_sequence_name():
    assert infer_platform_hint("final_challenge_ugv1") == "ugv"
    assert infer_platform_hint("long_corridor_rc") == "rc"
    assert infer_platform_hint("multi_floor_legrobot") == "legrobot"
    assert infer_platform_hint("laurel_cavern") == "unknown"


def test_summarize_timestamps_reports_intervals_and_order_issues():
    summary = summarize_timestamps([
        0,
        100_000_000,
        200_000_000,
        200_000_000,
        150_000_000,
    ])

    assert summary["min_timestamp"] == 0
    assert summary["max_timestamp"] == 200_000_000
    assert summary["duration_seconds"] == 0.2
    assert summary["median_interval_seconds"] == 0.1
    assert summary["sample_rate_hz"] == 10.0
    assert summary["duplicate_timestamp_count"] == 1
    assert summary["non_monotonic_timestamp_count"] == 1


def test_profile_imu_csv_returns_manifest_row_with_missing_counts():
    csv_text = """timestamp,q_x,q_y,q_z,q_w,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z
100,0.1,0.0,0.0,1.0,1.0,2.0,3.0,4.0,5.0,6.0
200,0.2,0.0,0.0,1.0,1.1,,3.1,4.1,5.1,6.1
300,0.3,0.0,0.0,1.0,1.2,2.2,3.2,4.2,5.2,6.2
"""

    profile = profile_imu_csv("long_corridor_rc", StringIO(csv_text))

    assert profile["sequence_name"] == "long_corridor_rc"
    assert profile["platform_hint"] == "rc"
    assert profile["row_count"] == 3
    assert profile["column_count"] == 11
    assert profile["columns"] == [
        "timestamp",
        "q_x",
        "q_y",
        "q_z",
        "q_w",
        "ang_vel_x",
        "ang_vel_y",
        "ang_vel_z",
        "lin_acc_x",
        "lin_acc_y",
        "lin_acc_z",
    ]
    assert profile["missing_value_counts"]["ang_vel_y"] == 1
    assert profile["missing_value_counts"]["timestamp"] == 0
    assert profile["duration_seconds"] == 2e-07
    assert profile["sample_rate_hz"] == 10_000_000.0
    assert profile["status"] == "ok"
