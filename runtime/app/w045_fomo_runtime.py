
"""Minimal frozen FOMO representation used by the w045 release runtime.

This is the inference-only architecture already exercised by the accepted
FOMO artifacts.  Keeping it separate avoids importing training/data modules
when the release runtime reconstructs the three frozen encoder states.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUnit(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, 3, padding=1, bias=True)
        self.norm = nn.InstanceNorm3d(
            output_channels, eps=1e-5, affine=True, momentum=0.1
        )
        self.activation = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(value)))


class ConvPair(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvUnit(input_channels, output_channels)
        self.conv2 = ConvUnit(output_channels, output_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(value))


class FomoUNetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_conv = ConvPair(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder_conv1 = ConvPair(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder_conv2 = ConvPair(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.encoder_conv3 = ConvPair(128, 256)
        self.pool4 = nn.MaxPool3d(2)
        self.encoder_conv4 = ConvPair(256, 512)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.in_conv(value)
        x1 = self.encoder_conv1(self.pool1(x0))
        x2 = self.encoder_conv2(self.pool2(x1))
        x3 = self.encoder_conv3(self.pool3(x2))
        x4 = self.encoder_conv4(self.pool4(x3))
        return [x0, x1, x2, x3, x4]


class FrozenFomoRepresentation(nn.Module):
    """Frozen encoder plus the accepted deterministic latent projections."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = FomoUNetEncoder()
        self.conv_mu_shared = nn.Conv3d(992, 256, 1)
        self.conv_mu_modality = nn.Conv3d(992, 256, 1)

    def encode_modality(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stages = self.encoder(value)
        size = stages[-1].shape[2:]
        pooled = torch.cat(
            [F.adaptive_avg_pool3d(stage, output_size=size) for stage in stages],
            dim=1,
        )
        return self.conv_mu_shared(pooled), self.conv_mu_modality(pooled)

    def forward(self, modalities: list[torch.Tensor]) -> torch.Tensor:
        if len(modalities) != 3:
            raise ValueError("FOMO runtime requires exactly T1w, FLAIR, and T2w")
        shared: list[torch.Tensor] = []
        specific: list[torch.Tensor] = []
        for modality in modalities:
            subject, sequence = self.encode_modality(modality)
            shared.append(subject)
            specific.append(sequence)
        latent = torch.cat([torch.stack(shared).mean(0), *specific], dim=1)
        return F.adaptive_avg_pool3d(latent, 1).flatten(1)
