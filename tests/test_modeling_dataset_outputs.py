import json
from pathlib import Path

from src.features.build_feature_dataset import build_feature_dataset
from src.modeling.modeling_dataset import ModelingWindowDataset, extract_window_array, load_fold_inputs
from tests.modeling_helpers import sample_clean_root, write_minimal_feature_artifacts


def test_load_fold_inputs_returns_expected_training_and_validation_views(tmp_path: Path):
    artifact_root = write_minimal_feature_artifacts(tmp_path)
    fold_inputs = load_fold_inputs(artifact_root, fold_name="fold_2", feature_set="raw", normalization="zscore")

    assert fold_inputs.fold_name == "fold_2"
    assert fold_inputs.train_sequences == ("final_challenge_ugv1", "urban_challenge_ugv1", "urban_challenge_ugv2")
    assert fold_inputs.validation_sequence == "final_challenge_ugv3"


def test_modeling_window_dataset_returns_tensor_and_metadata(tmp_path: Path):
    artifact_root = write_minimal_feature_artifacts(tmp_path)
    fold_inputs = load_fold_inputs(artifact_root, fold_name="fold_2", feature_set="raw", normalization="zscore")
    dataset = ModelingWindowDataset.from_fold_inputs(fold_inputs, split="train")

    sample = dataset[0]
    assert tuple(sample["inputs"].shape) == (150, 10)
    assert sample["metadata"]["fold_name"] == "fold_2"
    assert sample["metadata"]["split_name"] == "train"


def test_load_fold_inputs_reads_real_feature_builder_contract(tmp_path: Path):
    artifact_root = build_feature_dataset(
        clean_root=sample_clean_root(tmp_path),
        output_root=tmp_path / "features",
        dev_sequences=["seq_a", "seq_b", "seq_c", "seq_d"],
    )

    fold_inputs = load_fold_inputs(artifact_root, fold_name="fold_2", feature_set="raw", normalization="zscore")

    assert fold_inputs.train_sequences == ("seq_a", "seq_c", "seq_d")
    assert fold_inputs.validation_sequence == "seq_b"
    assert tuple(fold_inputs.feature_columns) == (
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
    )


def test_extract_window_array_applies_selected_normalizer(tmp_path: Path):
    artifact_root = write_minimal_feature_artifacts(tmp_path)
    robust_path = artifact_root / "normalizers" / "fold_2_raw_robust.json"
    payload = json.loads(robust_path.read_text())
    payload["center"] = {column: 9999.0 for column in payload["center"]}
    payload["scale"] = {column: 0.5 for column in payload["scale"]}
    robust_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    zscore_fold = load_fold_inputs(artifact_root, fold_name="fold_2", feature_set="raw", normalization="zscore")
    robust_fold = load_fold_inputs(artifact_root, fold_name="fold_2", feature_set="raw", normalization="robust")

    row = zscore_fold.windows_by_split["train"][0]
    frame = zscore_fold.feature_tables[row["sequence_name"]]
    zscore_window = extract_window_array(
        frame=frame,
        row=row,
        feature_columns=zscore_fold.feature_columns,
        normalizer=zscore_fold.normalizer,
    )
    robust_window = extract_window_array(
        frame=frame,
        row=row,
        feature_columns=robust_fold.feature_columns,
        normalizer=robust_fold.normalizer,
    )

    assert zscore_window[0, 0] != robust_window[0, 0]
