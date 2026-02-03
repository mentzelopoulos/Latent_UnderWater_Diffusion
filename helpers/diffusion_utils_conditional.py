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

import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
from helpers.pre_trained_autoencoder import encode, decode



device = "cuda" if torch.cuda.is_available() else "cpu"
autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

def extract(a, t, x_shape):
    return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

############# Variance Schedule

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps: int, low: float = 0.0001, high: float = 0.02):
    return torch.linspace(low, high, timesteps)


################## Forward Sample & sample_t


# Forward sample (noise addition)
def forward_sample(x_0, t, e, alphas_cumprod):
    alphas_cumprod = alphas_cumprod.to(t.device)
    sqrt_alphas_cumprod_t = extract(torch.sqrt(alphas_cumprod), t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(torch.sqrt(1. - alphas_cumprod), t, x_0.shape)
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * e
    v_t = sqrt_alphas_cumprod_t * e - sqrt_one_minus_alphas_cumprod_t * x_0
    return x_t, v_t

def sample_t(TOTAL_TIMESTEPS: int, batch_targets: torch.Tensor, bad_data:bool = False, batch_classes: torch.Tensor = None, bad_class_time_fraction: float = 0.5):
    
    if not bad_data:
        return torch.randint(0, TOTAL_TIMESTEPS, (batch_targets.shape[0],), device=device).long()
    
    t = torch.empty(batch_targets.shape[0], dtype=torch.long, device=device)
    is_bad = (batch_classes == 1)
    bad_high = max(1, int(bad_class_time_fraction * TOTAL_TIMESTEPS))  # randint(0, 0, ...) is invalid
    t[is_bad] = torch.randint(0, bad_high, (is_bad.sum(),), device=device)
    
    t[~is_bad] = torch.randint( # Sample t for good class from [0, TOTAL_TIMESTEPS)
        0, TOTAL_TIMESTEPS, ((~is_bad).sum(),), device=device
    )
    return t




############################# Reverse Diffusion

## Updated Reverse Process


@torch.no_grad()
@torch.amp.autocast(device_type=autocast_device)
def reverse_sample(model, x, x_c, t, t_index, cfg_guidance = False, cfg_scale = None):
            
    assert x.shape[1]==4 # Reverse sample on the latents
    model.eval()

    beta_t = model.beta_t.to(t.device)
    alphas = model.alphas.to(t.device)
    
    #alphas = 1. - beta_t # Currently stored in model
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x.shape)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    beta_tilde_t = beta_t * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    beta_tilde_t = extract(beta_tilde_t, t, x.shape)

    #alphas = 1. - beta_t # Currently stored in model
    #alphas_cumprod = torch.cumprod(alphas, axis=0) # Currently stored in model
    betas_t = extract(beta_t, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    # sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # Calculated in line 90
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    if cfg_guidance:
        v_cond = model(x, x_c, t)
        v_uncond = model(x, model.class_tokens.expand(x_c.size(0), -1, -1, -1), t)
        v = v_uncond + cfg_scale*(v_cond-v_uncond)
    else:
        v = model(x, x_c, t)
    
    x0_pred = sqrt_alphas_cumprod_t * x - sqrt_one_minus_alphas_cumprod_t * v # Predict x0 from velocity
    noise = (x - sqrt_alphas_cumprod_t * x0_pred) / sqrt_one_minus_alphas_cumprod_t # Compute predicted noise ε from x0_pred
    
    mu_tilde_t = sqrt_recip_alphas_t * (x - betas_t * noise/sqrt_one_minus_alphas_cumprod_t)
    e = torch.randn_like(x) if t.all() else torch.zeros_like(x) # Add noise if t is not 0
    return mu_tilde_t + torch.sqrt(beta_tilde_t) * e


@torch.no_grad()
@torch.amp.autocast(device_type=autocast_device)
def reverse_sample_ddim(model, x, x_c, t, t_next, guidance = False, cfg_scale = None):

    model.eval()
    #alphas and alphas_cumprod pre_computed globally
    alphas_cumprod = model.alphas_cumprod.to(t.device)
    
    alpha_t = extract(alphas_cumprod, t, x.shape) # Get ᾱ_t 
    alpha_next = extract(alphas_cumprod, t_next, x.shape) #and ᾱ_{t+1}

    sqrt_alpha_t = alpha_t.sqrt()
    sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
    sqrt_alpha_next = alpha_next.sqrt()
    sqrt_one_minus_alpha_next = (1 - alpha_next).sqrt()

    if guidance: # Classifier-free guidance
        v_cond = model(x, x_c, t)
        v_uncond = model(x, model.class_tokens.expand(x_c.size(0), -1, -1, -1), t)
        v = v_uncond + cfg_scale*(v_cond-v_uncond)    # (1+cfg_scale)*v_cond - cfg_scale*v_uncond
    else: # No guidance
        v = model(x, x_c, t)
        
    x0_pred = sqrt_alpha_t * x - sqrt_one_minus_alpha_t * v # Predict x0 from velocity
    pred_noise = (x - sqrt_alpha_t * x0_pred) / sqrt_one_minus_alpha_t # Compute predicted noise ε from x0_pred
    #pred_noise = (sqrt_one_minus_alpha_t * x0_pred + v) / sqrt_alpha_t
    x_next = sqrt_alpha_next * x0_pred + sqrt_one_minus_alpha_next * pred_noise # DDIM deterministic update step

    return x_next




@torch.no_grad()
@torch.amp.autocast(device_type=autocast_device)
def sample(model, autoencoder, corrupted_img=None, num_samples=16, channels=4, quick=False, num_timesteps=50, total_timesteps=1000, cfg_guidance=False, cfg_scale=3):
    model.eval()

    if corrupted_img is not None: # Conditional Generation
        assert corrupted_img.shape[1] == 3
        num_samples = corrupted_img.shape[0]
        x_c = encode(images = corrupted_img, autoencoder = autoencoder).to(device)
    else: # Unconditional Generation
        x_c = model.class_tokens.expand(num_samples, -1, -1, -1)

    shape = (num_samples, channels, x_c.shape[-2], x_c.shape[-1])
    img = torch.randn(shape, device=device)
    imgs = [img.cpu()]

    if not quick: # Full DDPM sampling
        times = torch.linspace(0, total_timesteps - 1, total_timesteps).long()
        if corrupted_img is None:
            cfg_guidance = False
        for i in tqdm(reversed(times), desc = "Generating images"):
            img = reverse_sample(model, img, x_c, torch.full((num_samples,), i, device=device, dtype=torch.long), i, cfg_guidance=cfg_guidance, cfg_scale=cfg_scale)
            imgs.append(img.cpu())
            
    else: # Fast DDIM sampling
        times = torch.linspace(total_timesteps - 1, 0, num_timesteps).long() #Sparse timesteps
        for idx in tqdm(range(len(times) - 1), desc = "Generating images"):
            t = times[idx]
            t_next = times[idx + 1]
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            t_next_batch = torch.full((num_samples,), t_next, device=device, dtype=torch.long)
            
            if not cfg_guidance:
                img = reverse_sample_ddim(model, img, x_c, t_batch, t_next_batch)
            else:
                img = reverse_sample_ddim(model, img, x_c, t_batch, t_next_batch, guidance = True, cfg_scale=cfg_scale)

            imgs.append(img.cpu())

    return imgs