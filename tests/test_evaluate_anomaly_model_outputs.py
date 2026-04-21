from pathlib import Path

import pandas as pd

from src.modeling.evaluate_anomaly_model import evaluate_experiment
from src.modeling.train_anomaly_model import train_experiment
from tests.modeling_helpers import write_minimal_feature_artifacts


def test_evaluate_experiment_writes_clean_and_injected_metrics(tmp_path: Path):
    artifact_root = write_minimal_feature_artifacts(tmp_path)
    output_dir = tmp_path / "modeling"
    train_result = train_experiment(
        artifact_root=artifact_root,
        output_dir=output_dir,
        architecture="lstm",
        feature_set="raw",
        normalization="zscore",
        fold_name="fold_2",
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=2,
        seed=7,
    )

    evaluate_result = evaluate_experiment(
        artifact_root=artifact_root,
        output_dir=output_dir,
        experiment_id=train_result.experiment_id,
        checkpoint_path=train_result.best_checkpoint_path,
        fold_name="fold_2",
        seed=7,
    )

    clean_metrics_path = train_result.run_dir / "clean_replay_metrics.csv"
    injected_metrics_path = train_result.run_dir / "injected_replay_metrics.csv"
    anomaly_breakdown_path = train_result.run_dir / "anomaly_breakdown.csv"
    assert clean_metrics_path.exists()
    assert injected_metrics_path.exists()
    assert anomaly_breakdown_path.exists()

    clean_metrics = pd.read_csv(clean_metrics_path)
    injected_metrics = pd.read_csv(injected_metrics_path)
    anomaly_breakdown = pd.read_csv(anomaly_breakdown_path)

    assert evaluate_result.clean_metrics_path == clean_metrics_path
    assert evaluate_result.injected_metrics_path == injected_metrics_path
    assert evaluate_result.anomaly_breakdown_path == anomaly_breakdown_path
    assert "clean_alert_rate" in clean_metrics.columns
    assert "event_detection_rate" in injected_metrics.columns
    assert "anomaly_type" in anomaly_breakdown.columns
    assert not anomaly_breakdown.empty
