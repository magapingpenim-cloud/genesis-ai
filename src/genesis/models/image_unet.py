"""Small text-conditioned image diffusion U-Net.

Research baseline only: trained from scratch, no pretrained weights.
Input images are RGB tensors in [-1, 1], shaped [B, 3, H, W].
Text is encoded with a tiny byte-level embedding learned jointly from captions.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

TEXT_LENGTH = 128
VOCAB_SIZE = 257  # 0 = padding, byte values are shifted by one


def encode_texts(texts: list[str], device: torch.device | str) -> torch.Tensor:
    """Encode UTF-8 text as padded byte IDs; tokenizer has no external vocabulary."""
    result = torch.zeros((len(texts), TEXT_LENGTH), dtype=torch.long)
    for row, text in enumerate(texts):
        encoded = list(text.encode("utf-8")[:TEXT_LENGTH])
        if encoded:
            result[row, :len(encoded)] = torch.tensor([value + 1 for value in encoded])
    return result.to(device)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)
    return F.pad(emb, (0, dim - emb.shape[1]))


class ResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(F.silu(condition)).chunk(2, dim=1)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return (x + h) / math.sqrt(2)


class ImageUNet(nn.Module):
    """Compact conditional U-Net for 32–128px image-diffusion experiments."""

    def __init__(self, base_channels: int = 64, text_dim: int = 128):
        super().__init__()
        c = base_channels
        self.text_embedding = nn.Embedding(VOCAB_SIZE, text_dim, padding_idx=0)
        self.text_projection = nn.Sequential(nn.Linear(text_dim, text_dim), nn.SiLU())
        self.time_projection = nn.Sequential(
            nn.Linear(c, c * 4), nn.SiLU(), nn.Linear(c * 4, c * 4)
        )
        cond_dim = c * 4
        self.text_to_condition = nn.Linear(text_dim, cond_dim)
        self.input = nn.Conv2d(3, c, 3, padding=1)
        self.down_block1 = ResBlock(c, cond_dim)
        self.downsample1 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)
        self.down_block2 = ResBlock(c * 2, cond_dim)
        self.downsample2 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)
        self.middle1 = ResBlock(c * 4, cond_dim)
        self.middle2 = ResBlock(c * 4, cond_dim)
        self.upsample2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.up_block2 = ResBlock(c * 4, cond_dim)
        self.upsample1 = nn.ConvTranspose2d(c * 4, c, 4, stride=2, padding=1)
        self.up_block1 = ResBlock(c * 2, cond_dim)
        self.output_norm = nn.GroupNorm(8 if c % 8 == 0 else 1, c)
        self.output = nn.Conv2d(c, 3, 3, padding=1)

    def forward(
        self,
        noisy_image: torch.Tensor,
        timesteps: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_image.ndim != 4 or noisy_image.shape[1] != 3:
            raise ValueError("noisy_image must have shape [B, 3, H, W]")
        if noisy_image.shape[-2] % 4 or noisy_image.shape[-1] % 4:
            raise ValueError("image height and width must be divisible by 4")
        if text_ids.ndim != 2 or text_ids.shape[0] != noisy_image.shape[0]:
            raise ValueError("text_ids must have shape [B, sequence_length]")
        mask = text_ids.ne(0).unsqueeze(-1)
        token_emb = self.text_embedding(text_ids)
        pooled = (token_emb * mask).sum(1) / mask.sum(1).clamp_min(1)
        text_cond = self.text_to_condition(self.text_projection(pooled))
        time_cond = self.time_projection(timestep_embedding(timesteps, self.time_projection[0].in_features))
        condition = time_cond + text_cond

        x0 = self.input(noisy_image)
        x1 = self.down_block1(x0, condition)
        x2 = self.down_block2(self.downsample1(x1), condition)
        x = self.middle1(self.downsample2(x2), condition)
        x = self.middle2(x, condition)
        x = self.upsample2(x)
        x = self.up_block2(torch.cat([x, x2], dim=1), condition)
        x = self.upsample1(x)
        x = self.up_block1(torch.cat([x, x1], dim=1), condition)
        return self.output(F.silu(self.output_norm(x)))
