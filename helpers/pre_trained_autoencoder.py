"""
Written and maintained by Andreas Mentzelopoulos
Copyright (c) 2025, Andreas Mentzelopoulos. All Rights Reserved.

This code is the exclusive property of Andreas Mentzelopoulos
All associated materials (data, models, scripts) are the
exclusive property of Andreas Mentzelopoulos and LOBSTgER.

This code may be used openly and freely for research and education purposes. 
No part of this code may be used, copied, distributed, or modified for commercial use, 
without the prior written consent of Andreas Mentzelopoulos.

For permission requests, contact: Andreas Mentzelopoulos, ament@mit.edu.
"""

import torch
from diffusers import AutoencoderKL

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_autoencoder(
    model_name: str = "CompVis/stable-diffusion-v1-4",
    subfolder: str = "vae",
    half_precision: bool = True
    
):
    """
    Load a pretrained Stable Diffusion autoencoder (VAE).

    Args:
        model_name (str): HuggingFace model repo id
        subfolder (str): subfolder containing the VAE weights

    Returns:
        autoencoder (AutoencoderKL): pretrained autoencoder, eval + frozen
    """

    autoencoder = (
        AutoencoderKL.from_pretrained(model_name, subfolder=subfolder)
        .eval()
        .requires_grad_(False)
        .to(device)
    )

    return autoencoder.half() if half_precision else autoencoder


@torch.no_grad()          # no gradients needed
@torch.amp.autocast(device_type=autocast_device)  # use mixed precision
def encode(images: torch.Tensor, autoencoder, batch_size: int = 16, scale_factor: float = 0.18215, to_cpu: bool = True):
    """
    Encode images into latents with VQ-GAN from Stable Diffusion.

    Args:
        images (B,3,H,W): torch.Tensor in [0,1]
        autoencoder: model with .encode()
        batch_size: split encoding to avoid OOM
        scale_factor: latent scaling (e.g. 0.18215)
        to_cpu: move latents to CPU

    Returns:
        Latents (torch.tensor) w/ shape (B,3,H//8,W//8)
    """
    assert images.ndim == 4 and images.shape[1] == 3, f"Expected batched tensor with 3 channels"

    images = 2 * images - 1  # normalize to [-1,1]
    out = []
    for i in range(0, images.shape[0], batch_size):
        z = autoencoder.encode(images[i:i+batch_size].to(device)).latent_dist.mean
        out.append(z.cpu() if to_cpu else z)

    return torch.cat(out, 0).float() * scale_factor if out else torch.empty(0)


@torch.no_grad()            # no gradients needed
@torch.amp.autocast(device_type=autocast_device)  # use mixed precision
def decode(z: torch.Tensor, autoencoder, batch_size: int = 16, scale_factor:float = 0.18215, to_cpu=True):
    """
    Decode latents into images with an autoencoder.

    Args:
        z (B,4,H',W'): latent tensor
        autoencoder: model with .decode()
        batch_size: split decoding to avoid OOM
        scale_factor: rescale latents before decoding
        to_cpu: move images to CPU

    Returns:
        Images (B,3,8*H',8*W') in [0,1]
    """
    assert z.ndim == 4 and z.shape[1] == 4, f"Expected batched tensor with 4 channels"

    decoded = []
    for i in range(0, z.shape[0], batch_size):
        z_batch = (z[i:i+batch_size].to(device)) / scale_factor
        x_batch = autoencoder.decode(z_batch).sample
        x_batch = torch.clamp((x_batch + 1.) / 2., 0., 1.)  # [-1,1] -> [0,1]
        decoded.append(x_batch.cpu() if to_cpu else x_batch)

    return torch.cat(decoded, 0).float()




