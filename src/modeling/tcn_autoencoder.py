from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padded = nn.functional.pad(inputs, (self.left_padding, 0))
        return self.conv(padded)


class ResidualTCNBlock(nn.Module):
    def __init__(self, *, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv2 = CausalConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.activation(self.conv1(inputs))
        outputs = self.activation(self.conv2(outputs))
        return outputs + residual


class TCNAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        channel_width: int,
        bottleneck_dim: int,
        dilations: tuple[int, ...],
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)
        self._downsample_stages = 2

        self.input_projection = weight_norm(nn.Conv1d(input_dim, channel_width, kernel_size=1))

        self.encoder_stage_1 = nn.ModuleList(
            [
                ResidualTCNBlock(channels=channel_width, kernel_size=kernel_size, dilation=dilation)
                for dilation in dilations[:2]
            ]
        )
        self.encoder_stage_2 = nn.ModuleList(
            [
                ResidualTCNBlock(channels=channel_width, kernel_size=kernel_size, dilation=dilation)
                for dilation in dilations[2:4]
            ]
        )
        self.encoder_stage_3 = nn.ModuleList(
            [
                ResidualTCNBlock(channels=channel_width, kernel_size=kernel_size, dilation=dilation)
                for dilation in dilations[4:]
            ]
        )

        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)

        self.to_bottleneck = weight_norm(nn.Conv1d(channel_width, bottleneck_dim, kernel_size=1))
        self.from_bottleneck = weight_norm(nn.Conv1d(bottleneck_dim, channel_width, kernel_size=1))

        self.decoder_stage_1 = nn.ModuleList(
            [
                ResidualTCNBlock(channels=channel_width, kernel_size=kernel_size, dilation=dilation)
                for dilation in (32, 16)
            ]
        )
        self.decoder_stage_2 = nn.ModuleList(
            [
                ResidualTCNBlock(channels=channel_width, kernel_size=kernel_size, dilation=dilation)
                for dilation in (8, 4)
            ]
        )
        self.decoder_stage_3 = nn.ModuleList(
            [
                ResidualTCNBlock(channels=channel_width, kernel_size=kernel_size, dilation=dilation)
                for dilation in (2, 1)
            ]
        )

        self.output_projection = weight_norm(nn.Conv1d(channel_width, input_dim, kernel_size=1))

    def compressed_length(self, sequence_length: int) -> int:
        length = sequence_length
        for _ in range(self._downsample_stages):
            length = (length + 1) // 2
        return length

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        _, sequence_length, _ = inputs.shape
        x = inputs.transpose(1, 2)
        encoded = self.input_projection(x)

        for block in self.encoder_stage_1:
            encoded = block(encoded)
        encoded = self.downsample(encoded)

        for block in self.encoder_stage_2:
            encoded = block(encoded)
        encoded = self.downsample(encoded)

        for block in self.encoder_stage_3:
            encoded = block(encoded)

        bottleneck = self.to_bottleneck(encoded)
        latent = bottleneck.mean(dim=-1)

        decoded = self.from_bottleneck(bottleneck)
        for block in self.decoder_stage_1:
            decoded = block(decoded)

        decoded = nn.functional.interpolate(decoded, scale_factor=2, mode="nearest")
        for block in self.decoder_stage_2:
            decoded = block(decoded)

        decoded = nn.functional.interpolate(decoded, scale_factor=2, mode="nearest")
        for block in self.decoder_stage_3:
            decoded = block(decoded)

        reconstruction = self.output_projection(decoded)
        reconstruction = self._match_length(reconstruction, sequence_length).transpose(1, 2)
        return {
            "reconstruction": reconstruction,
            "latent": latent,
            "bottleneck_sequence": bottleneck,
        }

    @staticmethod
    def _match_length(sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        current_length = sequence.size(-1)
        if current_length > target_length:
            return sequence[..., :target_length]
        if current_length < target_length:
            return nn.functional.pad(sequence, (0, target_length - current_length))
        return sequence
