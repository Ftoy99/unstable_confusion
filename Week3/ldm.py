from torch import nn

import torch
import torch.nn as nn
from unet import UNetModel


class LatentDiffusionModel(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super(LatentDiffusionModel, self).__init__()
        self.unet = UNetModel()
        self.timesteps = timesteps

        # Linear noise schedule (betas)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x_0, t):
        """
        Forward pass, applying the noise (forward diffusion process).
        x_0: Original input image
        t: Timesteps (batch of integers in range [0, timesteps])
        """
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        return x_t, noise

    def q_sample(self, x_0, t, noise):
        """
        Apply the forward diffusion process: q(x_t | x_0)
        Adds noise to the image based on timestep t.
        """
        return torch.sqrt(self.alpha_cumprod[t-1]) * x_0 + torch.sqrt(1 - self.alpha_cumprod[t-1]) * noise

    def reverse_diffusion(self, x_t, t):
        """
        Reverse process: denoise the image using the UNet.
        x_t: Noisy image at timestep t
        t: Timestep for reverse diffusion (denoising)
        """
        # Get timestep embedding (temb)
        temb = self.get_timestep_embedding(t)
        noise_pred = self.unet(x_t, temb)
        x_0_pred = self.predict_x0(x_t, t, noise_pred)
        return x_0_pred

    def predict_x0(self, x_t, t, noise_pred):
        """
        Predict the original image x_0 from noisy image x_t using the UNet output (predicted noise).
        """
        return (x_t - torch.sqrt(1 - self.alpha_cumprod[t]) * noise_pred) / torch.sqrt(self.alpha_cumprod[t])

    def get_timestep_embedding(self, t):
        """
        Generate timestep embeddings (temb) for conditioning.
        t: Timestep tensor (batch of integers).
        """
        # You can use a sinusoidal embedding or a learned embedding here
        half_dim = self.unet.q.weight.shape[0]
        emb = torch.log(self.alpha_cumprod[t] + 1e-6)
        emb = emb * (torch.pi * 2)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, 2*half_dim]
        return emb