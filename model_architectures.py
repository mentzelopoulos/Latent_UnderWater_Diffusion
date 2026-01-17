"""
Written and maintained by Andreas Mentzelopoulos
Copyright (c) 2025, Andreas Mentzelopoulos. All Rights Reserved.

This code is the exclusive property of Andreas Mentzelopoulos
All associated materials (data, models, scripts) are the
exclusive property of Andreas Mentzelopoulos and LOBSTgER.

No part of this code may be copied, distributed, modified, or used in any
form without the prior written consent of Andreas Mentzelopoulos.

For permission requests, contact: Andreas Mentzelopoulos, ament@mit.edu.
"""

from denoising_diffusion_pytorch import Unet
import torch
import torch.nn as nn
from helpers.diffusion_utils_conditional import linear_beta_schedule, cosine_beta_schedule

class UnconditionalUNet(nn.Module):
    def __init__(self, dim: int = 128, dim_mults = (1,2,4), channels: int = 4, beta_t: torch.Tensor = cosine_beta_schedule(timesteps = 1000), timesteps = 1000):
        super().__init__()

        ## Diffusion hyper-params
        self.beta_t = beta_t
        self.alphas = 1-beta_t
        self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)
        self.timesteps = timesteps

        ## U-Net hyperparams
        self.dim = dim
        self.dim_mults = dim_mults
        self.channels = channels
        
        self.unet = Unet(
            dim = dim,
            channels = channels,
            dim_mults = dim_mults,
            flash_attn = False,
        )

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor):
        return self.unet(x_noisy, t)  # Shape: same as x_noisy


class ConditionalUNet(nn.Module):
    def __init__(self, dim: int = 64, dim_mults = (1,2,4,8), channels: int = 4, image_size = (512//8, 768//8), num_classes = 1, beta_t: torch.Tensor = cosine_beta_schedule(timesteps = 1000), timesteps = 1000):
        super().__init__()

        ## Diffusion hyper-params
        self.beta_t = beta_t
        self.alphas = 1-beta_t
        self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)
        self.timesteps = timesteps

         ## U-Net hyperparams
        self.dim = dim
        self.dim_mults = dim_mults
        self.channels = channels

        # self.null_token (for unconditional generation)
        self.class_tokens = nn.Parameter(torch.randn(num_classes, channels, *image_size))
        
        self.unet = Unet(
            dim = dim,
            channels = channels*2,
            dim_mults = dim_mults,
            flash_attn = False,
        )

        self.proj = nn.Conv2d(2*channels, channels, kernel_size=1)

    def forward(self, x_noised: torch.Tensor, x_condition: torch.Tensor, t: torch.Tensor):
        unet_out = self.unet(torch.cat((x_noised, x_condition), dim=1), t)  # Shape: (B, 8, H, W)
        return self.proj(unet_out)  # Shape: (B, 4, H, W)
