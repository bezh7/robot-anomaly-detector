import torch

from src.modeling.lstm_autoencoder import LSTMAutoencoder


def test_lstm_autoencoder_reconstructs_full_window_shape():
    model = LSTMAutoencoder(input_dim=10, hidden_dim=64, latent_dim=32, num_layers=2)
    inputs = torch.randn(4, 150, 10)
    outputs = model(inputs)

    assert tuple(outputs["reconstruction"].shape) == (4, 150, 10)
    assert tuple(outputs["latent"].shape) == (4, 32)


def test_lstm_autoencoder_builds_learned_decoder_position_inputs():
    model = LSTMAutoencoder(input_dim=10, hidden_dim=64, latent_dim=32, num_layers=2)

    decoder_inputs = model.build_decoder_inputs(batch_size=2, seq_len=150, device=torch.device("cpu"))

    assert tuple(decoder_inputs.shape) == (2, 150, 64)
    assert not torch.allclose(decoder_inputs[:, 0, :], decoder_inputs[:, -1, :])
