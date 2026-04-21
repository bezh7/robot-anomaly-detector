from src.data.profiling import build_sequence_manifest


CSV_HEADER = (
    "timestamp,q_x,q_y,q_z,q_w,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z\n"
)


def test_build_sequence_manifest_profiles_multiple_s3_sequences():
    raw_prefix = "s3://example-bucket/raw/"
    responses = {
        (
            "aws",
            "s3",
            "ls",
            raw_prefix,
        ): "                           PRE corridor01/\n                           PRE multi_floor_legrobot/\n",
        (
            "aws",
            "s3",
            "cp",
            f"{raw_prefix}corridor01/imu_data.csv",
            "-",
        ): CSV_HEADER + "100,0,0,0,1,1,2,3,4,5,6\n200,0,0,0,1,1,2,3,4,5,6\n",
        (
            "aws",
            "s3",
            "cp",
            f"{raw_prefix}multi_floor_legrobot/imu_data.csv",
            "-",
        ): CSV_HEADER + "300,0,0,0,1,1,2,3,4,5,6\n400,0,0,0,1,1,2,3,4,5,6\n",
    }

    def runner(command: list[str]) -> str:
        return responses[tuple(command)]

    manifest = build_sequence_manifest(
        raw_prefix,
        runner=runner,
    )

    assert [row["sequence_name"] for row in manifest] == [
        "corridor01",
        "multi_floor_legrobot",
    ]
    assert manifest[0]["platform_hint"] == "unknown"
    assert manifest[1]["platform_hint"] == "legrobot"
    assert manifest[0]["row_count"] == 2
    assert manifest[1]["row_count"] == 2


import json

from src.data.profiling import write_manifest_outputs


def test_write_manifest_outputs_serializes_csv_and_json(tmp_path):
    manifest = [{
        "sequence_name": "corridor01",
        "platform_hint": "unknown",
        "row_count": 2,
        "column_count": 11,
        "columns": ["timestamp", "q_x"],
        "missing_value_counts": {"timestamp": 0, "q_x": 0},
        "status": "ok",
        "min_timestamp": 100,
        "max_timestamp": 200,
        "duration_seconds": 1e-7,
        "median_interval_seconds": 1e-7,
        "sample_rate_hz": 10000000.0,
        "duplicate_timestamp_count": 0,
        "non_monotonic_timestamp_count": 0,
    }]

    output_paths = write_manifest_outputs(manifest, tmp_path)

    csv_path = output_paths["csv"]
    json_path = output_paths["json"]
    assert csv_path.exists()
    assert json_path.exists()
    assert "corridor01" in csv_path.read_text()
    parsed_json = json.loads(json_path.read_text())
    assert parsed_json[0]["sequence_name"] == "corridor01"
