import torch

from src.modeling.tcn_autoencoder import TCNAutoencoder


def test_tcn_autoencoder_reconstructs_full_window_shape():
    model = TCNAutoencoder(
        input_dim=10,
        channel_width=64,
        bottleneck_dim=32,
        dilations=(1, 2, 4, 8, 16, 32),
    )
    inputs = torch.randn(4, 150, 10)
    outputs = model(inputs)

    assert tuple(outputs["reconstruction"].shape) == (4, 150, 10)
    assert tuple(outputs["latent"].shape) == (4, 32)
    assert outputs["bottleneck_sequence"].shape[0] == 4
    assert outputs["bottleneck_sequence"].shape[1] == 32
    assert outputs["bottleneck_sequence"].shape[2] < 150


def test_tcn_autoencoder_reports_receptive_field_covering_window():
    model = TCNAutoencoder(
        input_dim=10,
        channel_width=64,
        bottleneck_dim=32,
        dilations=(1, 2, 4, 8, 16, 32),
    )
    assert model.receptive_field >= 150


def test_tcn_autoencoder_reports_expected_compressed_length():
    model = TCNAutoencoder(
        input_dim=10,
        channel_width=64,
        bottleneck_dim=32,
        dilations=(1, 2, 4, 8, 16, 32),
    )
    assert model.compressed_length(150) == 38
