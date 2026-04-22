from pathlib import Path

from src.modeling.train_anomaly_model import train_final_experiment

from tests.modeling_helpers import write_minimal_feature_artifacts


def _write_final_model_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    feature_root = write_minimal_feature_artifacts(tmp_path, sequence_rows=520)
    modeling_root = tmp_path / "modeling_final"
    train_final_experiment(
        artifact_root=feature_root,
        output_dir=modeling_root,
        architecture="lstm",
        feature_set="raw_plus_derived",
        normalization="zscore",
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=1,
        seed=7,
    )
    return feature_root, modeling_root


def test_load_final_model_bundle_returns_checkpoint_and_config(tmp_path: Path) -> None:
    from src.modeling.replay_dashboard_data import load_final_model_bundle

    _feature_root, modeling_root = _write_final_model_artifacts(tmp_path)

    bundle = load_final_model_bundle(modeling_root)

    assert bundle.experiment_id == "lstm-raw_plus_derived-zscore-bs32-lr1e-03-seed7"
    assert bundle.run_dir.name == "full_dev"
    assert bundle.checkpoint_path.exists()
    assert bundle.config_path.exists()
    assert bundle.config["architecture"] == "lstm"
    assert bundle.config["feature_set"] == "raw_plus_derived"
    assert tuple(bundle.feature_columns) == tuple(bundle.config["feature_columns"])


def test_build_replay_timeline_returns_clean_and_demo_timelines(tmp_path: Path) -> None:
    from src.modeling.replay_dashboard_data import build_replay_timeline

    feature_root, modeling_root = _write_final_model_artifacts(tmp_path)

    clean_result = build_replay_timeline(
        feature_root=feature_root,
        model_root=modeling_root,
        sequence_name="final_challenge_ugv1",
        replay_mode="clean",
    )

    assert not clean_result.timeline.empty
    assert clean_result.anomaly_events == ()
    assert clean_result.replay_frame["sequence_name"].eq("final_challenge_ugv1").all()
    assert {
        "time_s",
        "score",
        "threshold",
        "alert_active",
        "top_group",
        "quaternion_score",
        "gyro_score",
        "accel_score",
        "anomaly_active",
    }.issubset(clean_result.timeline.columns)

    demo_result = build_replay_timeline(
        feature_root=feature_root,
        model_root=modeling_root,
        sequence_name="final_challenge_ugv1",
        replay_mode="demo",
        seed=11,
    )

    assert len(demo_result.anomaly_events) == 3
    assert {event["anomaly_type"] for event in demo_result.anomaly_events} == {
        "gyro_bias_drift",
        "accel_freeze",
        "clipping",
    }
    assert demo_result.timeline["anomaly_active"].any()
    assert {"target_group", "anomaly_type"}.issubset(demo_result.timeline.columns)
    assert "ang_vel_x" in demo_result.trace_columns
