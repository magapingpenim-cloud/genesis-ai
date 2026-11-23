"""A deliberately small research Video-DiT denoiser.

This is a baseline epsilon-prediction network, not a production video model.
It operates directly on low-resolution RGB clips shaped [B, C, T, H, W].
"""
from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal embeddings for integer diffusion timesteps."""
    half = dim // 2
    if half == 0:
        return values.float().unsqueeze(-1)
    scale = math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(
        torch.arange(half, device=values.device, dtype=torch.float32) * -scale
    )
    angles = values.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class VideoDiTDenoiser(nn.Module):
    """Small transformer that predicts noise in a video tensor.

    Input/output shape: [batch, channels, frames, height, width].
    The learned positional table imposes a maximum token count.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        max_tokens: int = 1024,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens

        self.input_projection = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)
        self.position = nn.Parameter(torch.zeros(1, max_tokens, hidden_dim))
        nn.init.trunc_normal_(self.position, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Conv3d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, video: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError("video must have shape [B, C, T, H, W]")
        batch, channels, frames, height, width = video.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, received {channels}")
        if timesteps.shape != (batch,):
            raise ValueError("timesteps must have shape [B]")

        features = self.input_projection(video)
        _, hidden, _, _, _ = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        token_count = tokens.shape[1]
        if token_count > self.max_tokens:
            raise ValueError(
                f"clip has {token_count} tokens, exceeding max_tokens={self.max_tokens}; "
                "use a smaller clip or increase the positional table"
            )

        time_embedding = sinusoidal_embedding(timesteps, hidden)
        tokens = tokens + self.position[:, :token_count] + self.time_mlp(time_embedding).unsqueeze(1)
        tokens = self.transformer(tokens)
        tokens = self.output_norm(tokens)

        features = tokens.transpose(1, 2).reshape(
            batch, hidden, frames, height, width
        )
        return self.output_projection(features)
