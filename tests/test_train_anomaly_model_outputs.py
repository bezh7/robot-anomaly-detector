from pathlib import Path

import pandas as pd
import torch

from src.modeling.lstm_autoencoder import LSTMAutoencoder
from src.modeling.train_anomaly_model import load_model_from_checkpoint, train_experiment, train_final_experiment
from tests.modeling_helpers import write_builder_feature_artifacts, write_minimal_feature_artifacts


def test_train_experiment_writes_checkpoint_and_history(tmp_path: Path):
    artifact_root = write_minimal_feature_artifacts(tmp_path)
    output_dir = tmp_path / "modeling"

    result = train_experiment(
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

    assert result.best_checkpoint_path.exists()
    assert (result.run_dir / "config.json").exists()
    history_path = result.run_dir / "train_history.csv"
    assert history_path.exists()

    history = pd.read_csv(history_path)
    assert list(history.columns) == ["epoch", "train_loss", "validation_loss", "is_best"]
    assert len(history) == 2


def test_load_model_from_checkpoint_uses_saved_model_config(tmp_path: Path):
    model = LSTMAutoencoder(
        input_dim=10,
        hidden_dim=64,
        latent_dim=32,
        num_layers=2,
        max_seq_len=37,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "architecture": "lstm",
            "model_config": {
                "input_dim": 10,
                "hidden_dim": 64,
                "latent_dim": 32,
                "num_layers": 2,
                "max_seq_len": 37,
            },
            "model_state_dict": model.state_dict(),
            "feature_columns": [],
            "experiment_id": "example",
            "fold_name": "fold_2",
        },
        checkpoint_path,
    )

    restored_model, checkpoint = load_model_from_checkpoint(checkpoint_path)

    assert checkpoint["model_config"]["max_seq_len"] == 37
    assert restored_model.max_seq_len == 37


def test_train_final_experiment_runs_on_feature_builder_artifacts(tmp_path: Path):
    artifact_root = write_builder_feature_artifacts(tmp_path)
    output_dir = tmp_path / "modeling"

    result = train_final_experiment(
        artifact_root=artifact_root,
        output_dir=output_dir,
        architecture="lstm",
        feature_set="raw",
        normalization="zscore",
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=1,
        seed=7,
    )

    assert result.best_checkpoint_path.exists()
    assert result.run_dir.name == "full_dev"
