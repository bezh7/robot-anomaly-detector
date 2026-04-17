import csv
import math
from io import StringIO
from statistics import fmean, median, pstdev
from typing import TextIO

from src.io_utils import Runner, list_s3_prefixes, read_s3_text
from src.profiling import infer_platform_hint, profile_imu_csv


DEFAULT_FEATURE_KEYS = [
    "gyro_norm_mean",
    "gyro_norm_std",
    "gyro_norm_p95",
    "accel_norm_mean",
    "accel_norm_std",
    "accel_norm_p95",
    "dynamic_accel_mean",
    "dynamic_accel_std",
    "dynamic_accel_p95",
    "gyro_delta_std",
    "accel_delta_std",
]


def _vector_norm(*components: float) -> float:
    return math.sqrt(sum(component * component for component in components))


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return sorted_values[rank]


def _safe_pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return pstdev(values)


def _zscore_vectors(signatures: list[dict[str, float]], feature_keys: list[str]) -> dict[str, list[float]]:
    means = {
        key: fmean(float(signature[key]) for signature in signatures)
        for key in feature_keys
    }
    stds = {
        key: _safe_pstdev([float(signature[key]) for signature in signatures])
        for key in feature_keys
    }

    vectors: dict[str, list[float]] = {}
    for signature in signatures:
        vectors[str(signature["sequence_name"])] = [
            (float(signature[key]) - means[key]) / (stds[key] or 1.0)
            for key in feature_keys
        ]
    return vectors


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dimensions = len(vectors[0])
    return [fmean(vector[index] for vector in vectors) for index in range(dimensions)]


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((l - r) ** 2 for l, r in zip(left, right)))


def compute_motion_signature(csv_stream: TextIO) -> dict[str, float]:
    reader = csv.DictReader(csv_stream, skipinitialspace=True)
    gyro_norms: list[float] = []
    accel_norms: list[float] = []

    for row in reader:
        gyro_norms.append(
            _vector_norm(
                float(row["ang_vel_x"]),
                float(row["ang_vel_y"]),
                float(row["ang_vel_z"]),
            )
        )
        accel_norms.append(
            _vector_norm(
                float(row["lin_acc_x"]),
                float(row["lin_acc_y"]),
                float(row["lin_acc_z"]),
            )
        )

    accel_baseline = median(accel_norms) if accel_norms else 0.0
    dynamic_accel = [abs(value - accel_baseline) for value in accel_norms]
    gyro_deltas = [current - previous for previous, current in zip(gyro_norms, gyro_norms[1:])]
    accel_deltas = [current - previous for previous, current in zip(accel_norms, accel_norms[1:])]

    return {
        "gyro_norm_mean": fmean(gyro_norms) if gyro_norms else 0.0,
        "gyro_norm_std": _safe_pstdev(gyro_norms),
        "gyro_norm_p95": _nearest_rank(gyro_norms, 0.95),
        "accel_norm_mean": fmean(accel_norms) if accel_norms else 0.0,
        "accel_norm_std": _safe_pstdev(accel_norms),
        "accel_norm_p95": _nearest_rank(accel_norms, 0.95),
        "dynamic_accel_mean": fmean(dynamic_accel) if dynamic_accel else 0.0,
        "dynamic_accel_std": _safe_pstdev(dynamic_accel),
        "dynamic_accel_p95": _nearest_rank(dynamic_accel, 0.95),
        "gyro_delta_std": _safe_pstdev(gyro_deltas),
        "accel_delta_std": _safe_pstdev(accel_deltas),
    }


def compare_candidate_to_reference_cohort(
    signatures: list[dict[str, float]],
    candidate_name: str,
    reference_names: list[str],
    feature_keys: list[str],
) -> dict[str, float | str | list[float]]:
    vectors = _zscore_vectors(signatures, feature_keys)
    reference_vectors = [vectors[name] for name in reference_names]
    reference_distances = []

    for name in reference_names:
        others = [vectors[other_name] for other_name in reference_names if other_name != name]
        reference_distances.append(_euclidean_distance(vectors[name], _centroid(others)))

    candidate_distance = _euclidean_distance(vectors[candidate_name], _centroid(reference_vectors))
    reference_distance_max = max(reference_distances) if reference_distances else 0.0

    return {
        "candidate_name": candidate_name,
        "candidate_distance": candidate_distance,
        "reference_distances": reference_distances,
        "reference_distance_max": reference_distance_max,
        "recommendation": "merge" if candidate_distance <= reference_distance_max else "separate",
    }


def build_motion_signature_manifest(s3_prefix: str, runner: Runner) -> list[dict[str, float | str]]:
    manifest: list[dict[str, float | str]] = []

    for sequence_name in list_s3_prefixes(s3_prefix, runner=runner):
        csv_text = read_s3_text(f"{s3_prefix}{sequence_name}/imu_data.csv", runner=runner)
        base_profile = profile_imu_csv(sequence_name, StringIO(csv_text))
        motion_signature = compute_motion_signature(StringIO(csv_text))
        manifest.append(
            {
                "sequence_name": sequence_name,
                "platform_hint": infer_platform_hint(sequence_name),
                "row_count": float(base_profile["row_count"]),
                "duration_seconds": float(base_profile["duration_seconds"]),
                "sample_rate_hz": float(base_profile["sample_rate_hz"] or 0.0),
                **motion_signature,
            }
        )

    return manifest


def analyze_rc_vs_ugv(
    signatures: list[dict[str, float | str]],
    feature_keys: list[str] | None = None,
) -> dict[str, object]:
    feature_keys = feature_keys or DEFAULT_FEATURE_KEYS
    rc_candidates = [
        str(signature["sequence_name"])
        for signature in signatures
        if signature["platform_hint"] == "rc"
    ]
    ugv_references = [
        str(signature["sequence_name"])
        for signature in signatures
        if signature["platform_hint"] == "ugv"
    ]

    if not rc_candidates:
        raise ValueError("No RC sequence found for comparison")
    if len(ugv_references) < 2:
        raise ValueError("Need at least two UGV sequences for cohort comparison")

    candidate_name = rc_candidates[0]
    comparison = compare_candidate_to_reference_cohort(
        signatures=[{key: value for key, value in signature.items() if key in set(feature_keys) | {"sequence_name"}} for signature in signatures],
        candidate_name=candidate_name,
        reference_names=ugv_references,
        feature_keys=feature_keys,
    )

    vectors = _zscore_vectors(
        [{key: value for key, value in signature.items() if key in set(feature_keys) | {"sequence_name"}} for signature in signatures],
        feature_keys,
    )
    candidate_vector = vectors[candidate_name]
    platform_lookup = {str(signature["sequence_name"]): str(signature["platform_hint"]) for signature in signatures}
    nearest_neighbors = sorted(
        [
            {
                "sequence_name": name,
                "platform_hint": platform_lookup[name],
                "distance": _euclidean_distance(candidate_vector, vector),
            }
            for name, vector in vectors.items()
            if name != candidate_name
        ],
        key=lambda item: item["distance"],
    )

    return {
        **comparison,
        "feature_keys": feature_keys,
        "reference_names": ugv_references,
        "nearest_neighbors": nearest_neighbors,
    }
