import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        max_seq_len: int = 150,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.latent_to_cell = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def build_decoder_inputs(
        self,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=device)
        embedded_positions = self.position_embedding(positions)
        return embedded_positions.unsqueeze(0).expand(batch_size, seq_len, self.hidden_dim)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = inputs.shape
        _, (hidden_state, _) = self.encoder(inputs)
        latent = self.to_latent(hidden_state[-1])

        decoder_hidden = self.latent_to_hidden(latent)
        decoder_hidden = decoder_hidden.view(batch_size, self.num_layers, self.hidden_dim)
        decoder_hidden = decoder_hidden.transpose(0, 1).contiguous()

        decoder_cell = self.latent_to_cell(latent)
        decoder_cell = decoder_cell.view(batch_size, self.num_layers, self.hidden_dim)
        decoder_cell = decoder_cell.transpose(0, 1).contiguous()

        decoder_inputs = self.build_decoder_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            device=inputs.device,
        )
        decoded, _ = self.decoder(decoder_inputs, (decoder_hidden, decoder_cell))
        reconstruction = self.output_projection(decoded)
        return {
            "reconstruction": reconstruction,
            "latent": latent,
        }
