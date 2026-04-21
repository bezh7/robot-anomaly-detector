from io import StringIO

from src.data.cohort_analysis import compute_motion_signature


def test_compute_motion_signature_uses_norm_based_features():
    csv_text = """timestamp,q_x,q_y,q_z,q_w,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z
0,0,0,0,1,0,0,0,0,0,10
1,0,0,0,1,3,4,0,6,8,0
2,0,0,0,1,6,8,0,0,6,8
"""

    signature = compute_motion_signature(StringIO(csv_text))

    assert signature["gyro_norm_mean"] == 5.0
    assert signature["gyro_norm_p95"] == 10.0
    assert signature["accel_norm_mean"] == 10.0
    assert signature["dynamic_accel_mean"] == 0.0
    assert signature["gyro_delta_std"] == 0.0


from src.data.cohort_analysis import compare_candidate_to_reference_cohort


def test_compare_candidate_to_reference_cohort_flags_close_rc_as_mergeable():
    signatures = [
        {"sequence_name": "ugv1", "feature_a": 0.0, "feature_b": 0.0},
        {"sequence_name": "ugv2", "feature_a": 0.1, "feature_b": 0.2},
        {"sequence_name": "ugv3", "feature_a": -0.1, "feature_b": 0.1},
        {"sequence_name": "rc1", "feature_a": 0.05, "feature_b": 0.05},
    ]

    report = compare_candidate_to_reference_cohort(
        signatures,
        candidate_name="rc1",
        reference_names=["ugv1", "ugv2", "ugv3"],
        feature_keys=["feature_a", "feature_b"],
    )

    assert report["candidate_name"] == "rc1"
    assert report["recommendation"] == "merge"
    assert report["candidate_distance"] <= report["reference_distance_max"]


CSV_HEADER = (
    "timestamp,q_x,q_y,q_z,q_w,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z\n"
)


def test_build_motion_signature_manifest_and_analyze_rc_vs_ugv():
    raw_prefix = "s3://example-bucket/raw/"
    responses = {
        ("aws", "s3", "ls", raw_prefix): (
            "                           PRE final_challenge_ugv1/\n"
            "                           PRE final_challenge_ugv2/\n"
            "                           PRE urban_challenge_ugv1/\n"
            "                           PRE long_corridor_rc/\n"
        ),
        ("aws", "s3", "cp", f"{raw_prefix}final_challenge_ugv1/imu_data.csv", "-"): (
            CSV_HEADER +
            "0,0,0,0,1,0,0,0,0,0,10\n" +
            "1,0,0,0,1,3,4,0,6,8,0\n" +
            "2,0,0,0,1,6,8,0,0,6,8\n"
        ),
        ("aws", "s3", "cp", f"{raw_prefix}final_challenge_ugv2/imu_data.csv", "-"): (
            CSV_HEADER +
            "0,0,0,0,1,0,0,0,0,0,10\n" +
            "1,0,0,0,1,2.4,3.2,0,6,8,0\n" +
            "2,0,0,0,1,4.8,6.4,0,0,6,8\n"
        ),
        ("aws", "s3", "cp", f"{raw_prefix}urban_challenge_ugv1/imu_data.csv", "-"): (
            CSV_HEADER +
            "0,0,0,0,1,0,0,0,0,0,10\n" +
            "1,0,0,0,1,3.6,4.8,0,6,8,0\n" +
            "2,0,0,0,1,7.2,9.6,0,0,6,8\n"
        ),
        ("aws", "s3", "cp", f"{raw_prefix}long_corridor_rc/imu_data.csv", "-"): (
            CSV_HEADER +
            "0,0,0,0,1,0,0,0,0,0,10\n" +
            "1,0,0,0,1,3.1,4.1,0,6,8,0\n" +
            "2,0,0,0,1,6.2,8.2,0,0,6,8\n"
        ),
    }

    def runner(command: list[str]) -> str:
        return responses[tuple(command)]

    from src.data.cohort_analysis import analyze_rc_vs_ugv, build_motion_signature_manifest

    signatures = build_motion_signature_manifest(
        raw_prefix,
        runner=runner,
    )
    report = analyze_rc_vs_ugv(signatures)

    assert len(signatures) == 4
    assert report["candidate_name"] == "long_corridor_rc"
    assert report["recommendation"] == "merge"
