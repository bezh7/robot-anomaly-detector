from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.features.fit_normalizers import fit_normalizer
from src.modeling.lstm_autoencoder import LSTMAutoencoder
from src.modeling.modeling_contract import build_experiment_id
from src.modeling.modeling_dataset import (
    ModelingWindowDataset,
    build_feature_columns,
    load_feature_tables,
    load_fold_inputs,
)
from src.modeling.tcn_autoencoder import TCNAutoencoder


@dataclass(frozen=True)
class TrainExperimentResult:
    experiment_id: str
    run_dir: Path
    best_checkpoint_path: Path
    best_validation_loss: float


def train_experiment(
    *,
    artifact_root: Path | str,
    output_dir: Path | str,
    architecture: str,
    feature_set: str,
    normalization: str,
    fold_name: str,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    seed: int,
    patience: int = 10,
    min_delta: float = 1e-4,
    device: str = "cpu",
) -> TrainExperimentResult:
    torch.manual_seed(seed)

    artifact_root = Path(artifact_root)
    output_dir = Path(output_dir)
    experiment_id = build_experiment_id(
        architecture=architecture,
        feature_set=feature_set,
        normalization=normalization,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    run_dir = output_dir / experiment_id / fold_name
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_inputs = load_fold_inputs(
        artifact_root,
        fold_name=fold_name,
        feature_set=feature_set,
        normalization=normalization,
    )
    train_loader, validation_loader = build_dataloaders(
        fold_inputs=fold_inputs,
        batch_size=batch_size,
    )
    model, model_config = build_model(
        architecture=architecture,
        input_dim=len(fold_inputs.feature_columns),
    )
    model.to(device)

    checkpoint_path = run_dir / "best_checkpoint.pt"
    _write_config(
        run_dir=run_dir,
        experiment_id=experiment_id,
        architecture=architecture,
        feature_set=feature_set,
        normalization=normalization,
        fold_name=fold_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        seed=seed,
        feature_columns=fold_inputs.feature_columns,
        model_config=model_config,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_training_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        validation_loss = _run_validation_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )
        is_best = validation_loss < (best_validation_loss - min_delta)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "is_best": is_best,
            }
        )

        if is_best:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "architecture": architecture,
                    "feature_columns": list(fold_inputs.feature_columns),
                    "experiment_id": experiment_id,
                    "fold_name": fold_name,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    pd.DataFrame(history_rows).to_csv(run_dir / "train_history.csv", index=False)
    return TrainExperimentResult(
        experiment_id=experiment_id,
        run_dir=run_dir,
        best_checkpoint_path=checkpoint_path,
        best_validation_loss=best_validation_loss,
    )


def build_dataloaders(*, fold_inputs, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_dataset = ModelingWindowDataset.from_fold_inputs(fold_inputs, split="train")
    validation_dataset = ModelingWindowDataset.from_fold_inputs(fold_inputs, split="validation")
    train_batch_size = max(1, min(batch_size, len(train_dataset)))
    validation_batch_size = max(1, min(batch_size, len(validation_dataset)))
    return (
        DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=_collate_batch),
        DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False, collate_fn=_collate_batch),
    )


def build_model(*, architecture: str, input_dim: int) -> tuple[nn.Module, dict[str, object]]:
    if architecture == "lstm":
        model_config = {
            "input_dim": input_dim,
            "hidden_dim": 64,
            "latent_dim": 32,
            "num_layers": 2,
            "max_seq_len": 150,
        }
        return _instantiate_model(architecture=architecture, model_config=model_config), model_config
    if architecture == "tcn":
        model_config = {
            "input_dim": input_dim,
            "channel_width": 64,
            "bottleneck_dim": 32,
            "dilations": (1, 2, 4, 8, 16, 32),
        }
        return _instantiate_model(architecture=architecture, model_config=model_config), model_config
    raise ValueError(f"unsupported architecture: {architecture}")


def load_model_from_checkpoint(checkpoint_path: Path | str, *, device: str = "cpu") -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model = _instantiate_model(
        architecture=str(checkpoint["architecture"]),
        model_config=dict(checkpoint["model_config"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _instantiate_model(*, architecture: str, model_config: dict[str, object]) -> nn.Module:
    if architecture == "lstm":
        return LSTMAutoencoder(**model_config)
    if architecture == "tcn":
        return TCNAutoencoder(**model_config)
    raise ValueError(f"unsupported architecture: {architecture}")


def train_final_experiment(
    *,
    artifact_root: Path | str,
    output_dir: Path | str,
    architecture: str,
    feature_set: str,
    normalization: str,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    seed: int,
    device: str = "cpu",
) -> TrainExperimentResult:
    torch.manual_seed(seed)

    artifact_root = Path(artifact_root)
    output_dir = Path(output_dir)
    experiment_id = build_experiment_id(
        architecture=architecture,
        feature_set=feature_set,
        normalization=normalization,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    run_dir = output_dir / experiment_id / "full_dev"
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_tables, feature_columns, windows, normalizer = _load_full_dev_windows(
        artifact_root=artifact_root,
        feature_set=feature_set,
        normalization=normalization,
    )
    train_windows, validation_windows = _split_final_training_windows(windows)
    train_dataset = ModelingWindowDataset(
        windows=train_windows,
        feature_tables=feature_tables,
        feature_columns=feature_columns,
        normalizer=normalizer,
    )
    validation_dataset = ModelingWindowDataset(
        windows=validation_windows,
        feature_tables=feature_tables,
        feature_columns=feature_columns,
        normalizer=normalizer,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, min(batch_size, len(train_dataset))),
        shuffle=True,
        collate_fn=_collate_batch,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=max(1, min(batch_size, len(validation_dataset))),
        shuffle=False,
        collate_fn=_collate_batch,
    )

    model, model_config = build_model(
        architecture=architecture,
        input_dim=len(feature_columns),
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()

    checkpoint_path = run_dir / "final_checkpoint.pt"
    _write_config(
        run_dir=run_dir,
        experiment_id=experiment_id,
        architecture=architecture,
        feature_set=feature_set,
        normalization=normalization,
        fold_name="full_dev",
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        seed=seed,
        feature_columns=feature_columns,
        model_config=model_config,
    )

    history_rows = []
    best_validation_loss = float("inf")
    for epoch in range(1, max_epochs + 1):
        train_loss = _run_training_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        validation_loss = _run_validation_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )
        best_validation_loss = min(best_validation_loss, validation_loss)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "is_best": validation_loss == best_validation_loss,
            }
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "architecture": architecture,
            "feature_columns": list(feature_columns),
            "experiment_id": experiment_id,
            "fold_name": "full_dev",
        },
        checkpoint_path,
    )
    pd.DataFrame(history_rows).to_csv(run_dir / "train_history.csv", index=False)
    return TrainExperimentResult(
        experiment_id=experiment_id,
        run_dir=run_dir,
        best_checkpoint_path=checkpoint_path,
        best_validation_loss=best_validation_loss,
    )


def _write_config(
    *,
    run_dir: Path,
    experiment_id: str,
    architecture: str,
    feature_set: str,
    normalization: str,
    fold_name: str,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    seed: int,
    feature_columns: tuple[str, ...],
    model_config: dict[str, object],
) -> None:
    payload = {
        "experiment_id": experiment_id,
        "architecture": architecture,
        "feature_set": feature_set,
        "normalization": normalization,
        "fold_name": fold_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "seed": seed,
        "feature_columns": list(feature_columns),
        "model_config": model_config,
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2))


def _collate_batch(samples: list[dict]) -> dict[str, object]:
    return {
        "inputs": torch.stack([sample["inputs"] for sample in samples], dim=0),
        "metadata": [sample["metadata"] for sample in samples],
    }


def _run_training_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    criterion,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs["reconstruction"], inputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item()) * inputs.size(0)
        total_examples += inputs.size(0)
    return total_loss / max(1, total_examples)


def _run_validation_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs["reconstruction"], inputs)
            total_loss += float(loss.item()) * inputs.size(0)
            total_examples += inputs.size(0)
    return total_loss / max(1, total_examples)


def _load_full_dev_windows(*, artifact_root: Path, feature_set: str, normalization: str):
    feature_tables = load_feature_tables(artifact_root)
    split_manifest = json.loads((artifact_root / "split_manifest.json").read_text())

    deduped = {}
    feature_columns = build_feature_columns(feature_set)
    for fold_index, _fold in enumerate(split_manifest["folds"], start=1):
        fold_name = f"fold_{fold_index}"
        path = artifact_root / "window_indices" / f"{fold_name}_train_{feature_set}_{normalization}.parquet"
        for row in pd.read_parquet(path).to_dict(orient="records"):
            key = (
                row["sequence_name"],
                int(row["start_row"]),
                int(row["end_row"]),
            )
            deduped[key] = row
    if not deduped:
        raise ValueError("no full-dev windows found")
    normalizer = fit_normalizer(
        feature_tables=feature_tables,
        training_sequences=sorted(feature_tables.keys()),
        feature_columns=list(feature_columns),
        mode=normalization,
    )
    windows = sorted(
        deduped.values(),
        key=lambda row: (row["sequence_name"], row["start_row"], row["end_row"]),
    )
    return feature_tables, feature_columns, windows, normalizer


def _split_final_training_windows(windows: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(windows) <= 1:
        return windows, windows
    validation_count = max(1, len(windows) // 10)
    return windows[:-validation_count], windows[-validation_count:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train anomaly autoencoders on fold or full-dev splits.")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fold-name")
    parser.add_argument("--final-train", action="store_true")
    args = parser.parse_args()

    if args.final_train:
        train_final_experiment(
            artifact_root=args.artifact_root,
            output_dir=args.output_root,
            architecture=args.architecture,
            feature_set=args.feature_set,
            normalization=args.normalization,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            seed=args.seed,
        )
        return

    if not args.fold_name:
        raise SystemExit("--fold-name is required unless --final-train is set")

    train_experiment(
        artifact_root=args.artifact_root,
        output_dir=args.output_root,
        architecture=args.architecture,
        feature_set=args.feature_set,
        normalization=args.normalization,
        fold_name=args.fold_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
