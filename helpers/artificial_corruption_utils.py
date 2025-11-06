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
from matplotlib import pyplot as plt

#################################### Synthetic Corruption (simulating underwater challenges) ##############

@torch.no_grad()
def corrupt_individually(x: torch.tensor, **args):
    assert x.ndim == 4 and x.shape[1] == 3 # Batched pixel space
    B, C, H, W = x.shape
    corrupted_images = []
    for img in x:
        img.unsqueeze_(0)
        corrupted_images.append(corrupt_fast(img, **args).squeeze())
    return torch.stack(corrupted_images)

@torch.cuda.amp.autocast()
@torch.no_grad()
def corrupt_fast(image: torch. tensor, device="cuda", add_bubbles: bool = True, num_bubbles: int = 100, max_bubble_size: int = 4, max_bubble_intensity: float =0.77,
            haze_intensity=None, p_haze: float = 0.75, p_color: float = 0.75, noise_mean: float = 0.0, noise_std: float = 0.05, color_shift_max: float = 0.15, motion_blur:bool = True, p_blur:float = 0.8,
            blur_kernel_size: list[int] =[4, 8, 16, 32, 64], blur_sigma: float = 5.0, vignetting_intensity: float = 0.2,
            apply_inpainting: bool = True, p_inpainting: float = 0.3):

    ## Expectes batched tensor of images

    if image.ndim != 4:
        print(image.ndim)
    assert image.ndim == 4 and image.shape[1] == 3, "Expected Batched tensor of images"
    
    image = image.to(device) if torch.cuda.is_available() else image.cpu()
    batch_size, channels, height, width = image.shape
    
    clean_image = image
    

    # Add bubbles
    if add_bubbles:
        num_bubbles = torch.randint(low=40, high=num_bubbles, size=(1,)).item()
        center_x = torch.randint(0, width, (num_bubbles,), device=device)
        center_y = torch.randint(0, height, (num_bubbles,), device=device)
        radius = torch.randint(1, max_bubble_size + 1, (num_bubbles,), device=device)
        intensity = torch.empty(num_bubbles, device=device).uniform_(0.69, max_bubble_intensity)
        y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        y, x = y.unsqueeze(0), x.unsqueeze(0)  # Shape (1, H, W)
        mask = ((x - center_x.view(-1, 1, 1)) ** 2 + (y - center_y.view(-1, 1, 1)) ** 2) <= radius.view(-1, 1, 1) ** 2
        mask = mask.any(dim=0).float()  # Collapse bubbles into one mask
        white_bubble = torch.ones_like(image) * intensity.view(-1, 1, 1, 1).max()
        image = image * (1 - mask) + white_bubble * mask
    
    # Add haze
    if torch.rand(1).item() < p_haze:
        if haze_intensity is None:
            haze_intensity = torch.rand(1).item() * 0.4 + 0.1
        haze = torch.full_like(image, fill_value=0.5)
        image = (1 - haze_intensity) * image + haze_intensity * haze

    if motion_blur and torch.rand(1).item() < p_blur:
        # Add motion blur
        blur_kernel_size = random.choice(blur_kernel_size)
        kernel = torch.arange(blur_kernel_size).float() - (blur_kernel_size // 2)
        kernel = torch.exp(-0.5 * (kernel / blur_sigma) ** 2)
        kernel /= kernel.sum()
        kernel_2d = torch.outer(kernel, kernel)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).expand(image.shape[1], 1, blur_kernel_size, blur_kernel_size).to(device)
        padding = blur_kernel_size // 2
        image_padded = F.pad(image, (padding, padding, padding, padding), mode='reflect')
        image = F.conv2d(image_padded, kernel_2d, padding=0, groups=image.shape[1])
        image = torch.clamp(image[:, :, :-1, :-1], 0.0, 1.0)

    # Add color shift
    if torch.rand(1).item() < p_color:
        shifts = torch.tensor([
            1 - torch.abs(torch.randn(1)) * 0.75*color_shift_max,  
            1.15 + torch.abs(torch.randn(1)) * 0.85*color_shift_max,  
            1 + torch.abs(torch.randn(1)) * color_shift_max * 0.35  
        ], device=device)
        image = torch.clamp(image * shifts.view(3, 1, 1), 0.0, 1.0)

    # Apply random missing patches for inpainting
    if apply_inpainting and torch.rand(1).item() < p_inpainting:
        for _ in range(5):
            h = torch.randint(16, 256, (1,)).item()
            w = torch.randint(16, 256, (1,)).item()
            top = torch.randint(0, height - h, (1,)).item()
            left = torch.randint(0, width - w, (1,)).item()
            image[:, :, top:top + h, left:left + w] = torch.randn_like(image[:, :, top:top + h, left:left + w])  # could also be 0 or random noise

    '''
    # Add vignetting
    x, y = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    dist = torch.sqrt((x - height // 2) ** 2 + (y - width // 2) ** 2)
    max_dist = torch.max(dist)
    vignette_mask = 1 - (dist / max_dist) * vignetting_intensity
    vignette_mask = vignette_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, height, width)
    image = image * vignette_mask

    # Add Gaussian noise
    noise = torch.normal(mean=noise_mean, std=noise_std, size=image.shape, device=device)
    image = torch.clamp(image + noise, 0.0, 1.0)

    # Darken
    gamma = torch.rand(1, device=device) * 0.2 + 0.8
    image = torch.clamp(image * gamma, 0.0, 1.0)
    '''

    # --- Blending with original clean image ---
    # Create a binary mask of random patches (1 = corrupted, 0 = keep original)
    num_patches = torch.randint(5, 15, (1,)).item()  # choose how many patches per image
    corruption_mask = torch.zeros((batch_size, 1, height, width), device=device)

    for _ in range(num_patches):
        min_frac, max_frac = 0.25, 0.60  # patch covers 5–25% of dimension
        h = torch.randint(int(height*min_frac), int(height*max_frac), (1,)).item()
        w = torch.randint(int(width*min_frac), int(width*max_frac), (1,)).item()
        top = torch.randint(0, height - h, (1,)).item()
        left = torch.randint(0, width - w, (1,)).item()
        corruption_mask[:, :, top:top+h, left:left+w] = 1.0

    # Expand mask to all channels
    corruption_mask = corruption_mask.expand(-1, channels, -1, -1)

    # Blend clean and corrupted image
    image = image * corruption_mask + clean_image * (1 - corruption_mask)

    #plt.imshow(image.squeeze().detach().cpu().numpy().transpose(1,2,0))
    #plt.show()
    #assert False if torch.randn(1) > 2.5 else True
    
    return image.float().cpu()

def blend(image: torch.tensor, clean_image:torch.tensor, min_frac = 0.15, max_frac = 0.6):
    
    num_patches = torch.randint(5, 15, (1,)).item()
    for _ in range(num_patches):
        # min_frac, max_frac = 0.25, 0.60  # patch covers 5–25% of dimension
        h = torch.randint(int(height*min_frac), int(height*max_frac), (1,)).item()
        w = torch.randint(int(width*min_frac), int(width*max_frac), (1,)).item()
        top = torch.randint(0, height - h, (1,)).item()
        left = torch.randint(0, width - w, (1,)).item()
        corruption_mask[:, :, top:top+h, left:left+w] = 1.0

    corruption_mask = corruption_mask.expand(-1, image.shape[1], -1, -1)

    return image * corruption_mask + clean_image * (1 - corruption_mask)
    


##################################### Individual Corruptions (below) #####################################

@torch.no_grad()
def add_haze(image: torch.tensor, intensity=None):
    """
    Simulate haze/fog effect by blending the image with a gray layer.
    intensity: float, strength of the haze (0=no haze, 1=full haze)
    """
    if intensity is None:
        intensity = torch.rand(1).item() * 0.4 + 0.3  # Random haze between 0.3 and 0.7
    haze = torch.full_like(image, 0.5)  # Gray haze
    return (1 - intensity) * image + intensity * haze

@torch.no_grad()
def add_gaussian_noise(image: torch.tensor, mean=0.0, std=0.05):
    """
    Add Gaussian noise to the image.
    mean: mean of noise
    std: standard deviation of noise
    """
    noise = torch.normal(mean=mean, std=std, size=image.shape).to(image.device)
    return torch.clamp(image + noise, 0.0, 1.0)  # Clamp to valid [0,1] range

@torch.no_grad()
def add_color_shift(image:torch.tensor, max_shift:float =0.25):
    """
    Randomly adjust color channels to simulate color cast or lighting changes.
    max_shift: maximum relative change per channel
    """
    shifts = torch.tensor([
        1 - torch.abs(torch.randn(1)) * max_shift,  # Red: reduce
        1 + torch.abs(torch.randn(1)) * max_shift,  # Green: increase
        1 + torch.abs(torch.randn(1)) * max_shift * 0.75  # Blue: increase but less
    ], device=image.device)
    return torch.clamp(image * shifts.view(3, 1, 1), 0.0, 1.0)

@torch.no_grad()
def add_motion_blur(image:torch.tensor, kernel_size: int =8, sigma: float =5.0):
    """
    Apply approximate motion blur using a Gaussian kernel.
    kernel_size: size of the blur kernel
    sigma: spread of the Gaussian kernel
    """
    # Create 1D Gaussian kernel
    kernel = torch.arange(kernel_size).float() - (kernel_size // 2)
    kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
    kernel /= kernel.sum()  # Normalize
    # Make 2D kernel for convolution
    kernel_2d = torch.outer(kernel, kernel).unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.expand(image.shape[1], 1, kernel_size, kernel_size).to(image.device)
    # Pad image to preserve size
    padding = kernel_size // 2
    image_padded = F.pad(image, (padding, padding, padding, padding), mode='reflect')
    # Convolve with kernel per channel
    blurred_image = F.conv2d(image_padded, kernel_2d, padding=0, groups=image.shape[1])
    return torch.clamp(blurred_image[:, :, :-1, :-1], 0.0, 1.0)

@torch.no_grad()
def add_vignetting(image: torch.tensor, intensity:float = 0.2):
    """
    Darken edges of the image to simulate lens vignetting.
    intensity: strength of the effect
    """
    B, C, H, W = image.shape
    # Create distance mask from image center
    x, y = torch.meshgrid(torch.arange(H, device=image.device), torch.arange(W, device=image.device), indexing='ij')
    dist = torch.sqrt((x - H // 2) ** 2 + (y - W // 2) ** 2)
    vignette_mask = 1 - (dist / dist.max()) * intensity
    vignette_mask = vignette_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
    return image * vignette_mask

@torch.no_grad()
def add_bubbles(image: torch.tensor, num_bubbles: int = 100, max_size: int =4, max_intensity: float = 0.77):
    """
    Add white circular bubbles at random positions.
    num_bubbles: max number of bubbles per image
    max_size: maximum radius of bubbles
    max_intensity: brightness of bubbles
    """
    B, C, H, W = image.shape
    out = image.clone()
    num_bubbles = torch.randint(0, num_bubbles, (1,)).item()
    for _ in range(num_bubbles):
        cx, cy = random.randint(0, W-1), random.randint(0, H-1)
        radius = random.randint(1, max_size)
        # Create circular mask
        y, x = torch.meshgrid(torch.arange(H, device=image.device), torch.arange(W, device=image.device), indexing='ij')
        mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
        intensity = random.uniform(0.69, max_intensity)
        for b in range(B):
            out[b] = out[b] * (~mask).float() + intensity * mask.float()
    return out

@torch.no_grad()
def darken(image:torch.tensor):
    """
    Randomly darken the image to simulate low-light conditions.
    """
    gamma = torch.rand(1, device=image.device) * 0.15 + 0.85  # scale between 0.85-1.0
    return torch.clamp(image * gamma, 0.0, 1.0)

@torch.no_grad()
def corrupt(image: torch.tensor, device: str ="cuda"):
    """
    Apply a full sequence of image corruptions.
    Moves image to device, then applies:
        bubbles, haze, noise, color shift, motion blur, vignetting, darkening
    """
    image = image.to(device)
    image = add_bubbles(image)
    image = add_haze(image)
    image = add_gaussian_noise(image)
    image = add_color_shift(image)
    image = add_motion_blur(image)
    image = add_vignetting(image)
    image = darken(image)
    return image
