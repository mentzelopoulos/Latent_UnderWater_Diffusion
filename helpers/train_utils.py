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

import os
import torch
import gc
import math
import random
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional

from helpers.read_data_tools import latentDataset, save_latentDataset, augment_batch, beautify_batch
from helpers.artificial_corruption_utils import corrupt_fast
from helpers.pre_trained_autoencoder import encode

device = "cuda" if torch.cuda.is_available() else "cpu"
autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

############# Custom LR Scheduler 

def flat_then_decay_w_warmup(optimizer, num_epochs, flat_until = 0.9 , warmup_steps=250):

    flat_steps = int(flat_until * num_epochs)
    decay_steps = max(1, num_epochs - flat_steps)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        if step < flat_steps:
            return 1.0
        progress = (step - flat_steps) / decay_steps
        return 1.0 / (1.0 + math.exp(12 * (progress - 0.5)))
    
    return LambdaLR(optimizer, lr_lambda)

############# Checkpointing

def load_model(
    model: torch.nn.Module,
    checkpoint_path: str = "model_checkpoints/demo_model.pth"):
    
    """
    Loads model from checkpoint if available.

    Returns:
       None
    """
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['ema_params'], strict=False)
    else:
        print("Cannot load_model, checkpoint does not exist")

    return None

def read_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer = None,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    dataset_path: str = 'images/data_checkpoints/train_latentDataset.pt',
    checkpoint_path: str = "model_checkpoints/demo_model.pth",
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    reset_lr = False
) -> Tuple[int, Dict[str, torch.Tensor], List[float], DataLoader]:
    
    """
    Resumes training from checkpoint if available, otherwise initializes training.

    Returns:
        epoch_start: Starting epoch
        ema_params: EMA parameters
        losses_total: List of total losses
        train_loader: DataLoader for training
    """
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch_start = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model state_dict loaded")
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state_dict loaded")
        else:
            print("No Optimizer state_dict loaded")
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state_dict loaded")
        else:
            print("No scheduler state_dict loaded")
            
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("Gradient scaler state_dict loaded")
        else:
            print("No gradient scaler state_dict loaded")
        
        ema_params = checkpoint['ema_params']
        for name, param in model.named_parameters():
            if param.requires_grad and name in ema_params:
                ema_params[name] = ema_params[name].to(device)   
        losses_total = checkpoint['losses_total']

        # Reset optimizer learning rate
        if reset_lr:
            print("Warning: Re-setting learning rate to: ", learning_rate) 
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

    else:
        epoch_start = -1
        losses_total = []
        ema_params = {name: param.detach().clone().to(device) for name, param in model.named_parameters()}
        print("Warning: Checkpoint does not exist, starting from scratch!")

    
    # Load dataset
    try:
        print("Loaded train_loader")
        data = torch.load(dataset_path, map_location="cpu")
        dataset = latentDataset(
            conditions=data['conditions'],
            targets=data['targets'],
            classes=data['classes'],
            raw_latents = data['raw_latents']
        )
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except:
        train_loader = None
        print("Train loader not found and not loaded")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model number of parameters: {total_params}")

    return epoch_start, ema_params, losses_total, train_loader 


############## Model updating

@torch.no_grad()
def update_ema(model, ema_params: dict, ema_decay: float):
    for name, param in model.named_parameters():
        if param.requires_grad and name in ema_params:
            ema_params[name].mul_(ema_decay).add_(param.detach(), alpha=1 - ema_decay)
    return 

############## Memory cleanup

def clean():
    gc.collect()
    torch.cuda.empty_cache()
    return 


############ Result logging

def print_training_logs(epoch: int, losses_total: list, optimizer, save_path = "./results/"):
    plt.clf()
    plt.plot(losses_total, label="Training")
    plt.ylim(0.01, 0.4)
    plt.grid()
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(save_path + "Loss.png")

    learning_rates = [param_group['lr'] for param_group in optimizer.param_groups]
    print(f"Epoch: {epoch+1} learning rate: {learning_rates[-1]}")
    return 



############### Checkpointing

def write_checkpoint(epoch: int, model: nn.Module, optimizer, ema_params: dict, losses_total: list, train_loader, scheduler = None, scaler = None,
                model_name: str = "demo_model", model_save_path: str = "./model_checkpoints/", train_data_save_path: str = "train_images/data_checkpoints/"):
    
    if scheduler is None:
        print("Warning: Scheduler not received and not being saved")
    if scaler is None:
        print("Warning: Scaler not received and not being saved, are you using MP?")
        
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'ema_params': ema_params,
        'losses_total': losses_total
    }, model_save_path + model_name + ".pth")
    
    print(f"Model saved, epoch: {epoch+1}")
    
    save_latentDataset(train_loader.dataset, save_name = "train_latentDataset")
    return 


################# Update Dataloader

@torch.no_grad()
@torch.amp.autocast(device_type=autocast_device)
def update_dataloader(train_loader, train_imageDataset, autoencoder, N=200, beautify=True, num_augmentations_per_image=1):
    """
    Update a subset of a LatentDataset inside a DataLoader by re-encoding images.

    Args:
        train_loader: DataLoader with a LatentDataset
        train_images: list or tensor of original images
        train_labels: list/tensor of labels
        N: number of samples to update
        beautify: whether to apply beautification
        num_augmentations_per_image: number of augmentations per sample
    """
    dataset = train_loader.dataset
    N = min(N, len(dataset))
    indices = random.sample(range(len(dataset)), N)  # pick N unique indices

    for i in tqdm(indices, desc=f"Updating {N} Samples"):
        image = train_imageDataset.images[i].unsqueeze(0).to(device)

        for _ in range(num_augmentations_per_image):
            image_aug = augment_batch(image)

            # Encode corrupted/beautified versions
            if beautify:
                combined = torch.cat((
                    corrupt_fast(image_aug, apply_inpainting=True),
                    beautify_batch(image_aug)
                ), dim=0)
            else:
                combined = torch.cat((
                    corrupt_fast(image_aug),
                    image_aug
                ), dim=0)

            encodings = encode(images = combined, autoencoder = autoencoder)

            # Update dataset in-place
            dataset.conditions[i] = encodings[0].detach()
            dataset.targets[i] = encodings[1].detach()

    return train_loader


########################## Sample and Save

@torch.no_grad()
def sample_and_save_images(model, epoch, corrupted_img: torch.Tensor = None, num_samples = 8, save_location = "./results/Training_samples/"):

    save_dir = save_location + f'{epoch}'
    os.makedirs(save_dir, exist_ok=True)
    
    samples = sample(model, corrupted_img, num_samples)
    samples = samples[-1]

    for i in range(num_samples):
        r_samples = torch.tensor(samples[[i]])
        r_samples = decode(r_samples)
    
        tosave_img = r_samples.squeeze().detach().cpu().numpy()
        tosave_img = tosave_img.transpose(1,2,0)
        tosave_img = (tosave_img*255).astype('uint8')
        tosave_img = Image.fromarray(tosave_img)
        tosave_img.save(os.path.join(save_dir, f'myJelly{i}.png'))

    model.train() # Ensure model returns to training mode
    return None




