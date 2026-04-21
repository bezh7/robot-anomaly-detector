import csv
import json
from io import StringIO
from pathlib import Path
from statistics import median
from typing import TextIO

from src.common.io_utils import Runner, list_s3_prefixes, read_s3_text


NANOSECONDS_PER_SECOND = 1_000_000_000


def infer_platform_hint(sequence_name: str) -> str:
    normalized_name = sequence_name.lower()

    if "ugv" in normalized_name:
        return "ugv"
    if "legrobot" in normalized_name:
        return "legrobot"
    if normalized_name.endswith("_rc") or "_rc_" in normalized_name or normalized_name.startswith("rc"):
        return "rc"
    return "unknown"


def summarize_timestamps(timestamps: list[int]) -> dict[str, float | int | None]:
    if not timestamps:
        return {
            "min_timestamp": None,
            "max_timestamp": None,
            "duration_seconds": 0.0,
            "median_interval_seconds": None,
            "sample_rate_hz": None,
            "duplicate_timestamp_count": 0,
            "non_monotonic_timestamp_count": 0,
        }

    deltas = [current - previous for previous, current in zip(timestamps, timestamps[1:])]
    duplicate_timestamp_count = sum(1 for delta in deltas if delta == 0)
    non_monotonic_timestamp_count = sum(1 for delta in deltas if delta < 0)
    positive_deltas = [delta for delta in deltas if delta > 0]

    median_interval_seconds = None
    sample_rate_hz = None
    if positive_deltas:
        median_interval_seconds = median(positive_deltas) / NANOSECONDS_PER_SECOND
        sample_rate_hz = 1.0 / median_interval_seconds

    return {
        "min_timestamp": min(timestamps),
        "max_timestamp": max(timestamps),
        "duration_seconds": (max(timestamps) - min(timestamps)) / NANOSECONDS_PER_SECOND,
        "median_interval_seconds": median_interval_seconds,
        "sample_rate_hz": sample_rate_hz,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "non_monotonic_timestamp_count": non_monotonic_timestamp_count,
    }


def profile_imu_csv(sequence_name: str, csv_stream: TextIO) -> dict[str, object]:
    reader = csv.DictReader(csv_stream, skipinitialspace=True)
    columns = reader.fieldnames or []
    missing_value_counts = {column: 0 for column in columns}
    timestamps: list[int] = []
    row_count = 0

    for row in reader:
        row_count += 1
        for column in columns:
            value = row.get(column)
            if value is None or value == "":
                missing_value_counts[column] += 1

        timestamp_value = row.get("timestamp")
        if timestamp_value not in (None, ""):
            timestamps.append(int(timestamp_value))

    timestamp_summary = summarize_timestamps(timestamps)

    return {
        "sequence_name": sequence_name,
        "platform_hint": infer_platform_hint(sequence_name),
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "missing_value_counts": missing_value_counts,
        "status": "ok" if row_count > 0 else "empty",
        **timestamp_summary,
    }


def build_sequence_manifest(s3_prefix: str, runner: Runner) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []

    for sequence_name in list_s3_prefixes(s3_prefix, runner=runner):
        csv_text = read_s3_text(f"{s3_prefix}{sequence_name}/imu_data.csv", runner=runner)
        manifest.append(profile_imu_csv(sequence_name, StringIO(csv_text)))

    return manifest


def write_manifest_outputs(manifest: list[dict[str, object]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sequence_manifest.csv"
    json_path = output_dir / "sequence_manifest.json"

    json_path.write_text(json.dumps(manifest, indent=2))

    if manifest:
        fieldnames = list(manifest[0].keys())
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in manifest:
                serialized_row = {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
                writer.writerow(serialized_row)
    else:
        csv_path.write_text("")

    return {"csv": csv_path, "json": json_path}
