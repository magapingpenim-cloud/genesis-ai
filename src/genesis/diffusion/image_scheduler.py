"""DDPM scheduler for image diffusion."""
from __future__ import annotations
import torch
from torch import nn


class ImageDDPM(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be >= 2")
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
        self.timesteps = timesteps
        for name, value in {
            "betas": betas,
            "alphas": alphas,
            "alpha_bars": alpha_bars,
            "alpha_bars_prev": alpha_bars_prev,
            "posterior_variance": betas * (1 - alpha_bars_prev) / (1 - alpha_bars),
        }.items():
            self.register_buffer(name, value)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha_bars[t].view(-1, 1, 1, 1).to(clean.dtype)
        return a.sqrt() * clean + (1 - a).sqrt() * noise

    @torch.no_grad()
    def step(self, model, image: torch.Tensor, t: int, text_ids: torch.Tensor,
             guidance_scale: float = 4.0, null_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch = image.shape[0]
        times = torch.full((batch,), t, device=image.device, dtype=torch.long)
        eps_cond = model(image, times, text_ids)
        if null_ids is not None and guidance_scale != 1.0:
            eps_uncond = model(image, times, null_ids)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond

        alpha = self.alphas[t].to(image.dtype)
        alpha_bar = self.alpha_bars[t].to(image.dtype)
        beta = self.betas[t].to(image.dtype)
        mean = (image - beta / torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha)
        if t == 0:
            return mean
        noise = torch.randn_like(image)
        variance = self.posterior_variance[t].to(image.dtype)
        return mean + variance.sqrt() * noise

    @torch.no_grad()
    def ddim_step(self, model, image: torch.Tensor, t: int, previous_t: int,
                  text_ids: torch.Tensor, guidance_scale: float = 4.0,
                  null_ids: torch.Tensor | None = None) -> torch.Tensor:
        """DDIM update supporting skipped timesteps for faster sampling."""
        batch = image.shape[0]
        times = torch.full((batch,), t, device=image.device, dtype=torch.long)
        eps = model(image, times, text_ids)
        if null_ids is not None and guidance_scale != 1.0:
            uncond = model(image, times, null_ids)
            eps = uncond + guidance_scale * (eps - uncond)
        alpha_t = self.alpha_bars[t].to(image.dtype)
        alpha_prev = (self.alpha_bars[previous_t].to(image.dtype)
                      if previous_t >= 0 else torch.ones((), device=image.device, dtype=image.dtype))
        predicted_clean = (image - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
        return alpha_prev.sqrt() * predicted_clean + (1 - alpha_prev).sqrt() * eps
