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

import argparse
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from helpers.read_data_tools import read_data, imageDataset, latentDataset, build_latentDataset, load_latentDataset, load_imageDataset
from helpers.train_utils import read_checkpoint, clean, print_training_logs, write_checkpoint, update_ema, flat_then_decay_w_warmup, update_dataloader
from helpers.pre_trained_autoencoder import load_autoencoder
from helpers.diffusion_utils_conditional import forward_sample, cosine_beta_schedule, sample_t
from helpers.artificial_corruption_utils import corrupt_fast
from model_architectures import UnconditionalUNet, ConditionalUNet


def parse_args():
    """
    Parse command-line arguments for training hyperparameters.
    """
    parser = argparse.ArgumentParser(description='Train Latent Underwater Diffusion Model')
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=1e-2,
                       help='Learning rate (default: 1e-2)')
    parser.add_argument('--num-epochs', type=int, default=1000,
                       help='Number of training epochs (default: 1000)')
    parser.add_argument('--total-timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps (default: 1000)')
    parser.add_argument('--cfg-dropout', type=float, default=0.25,
                       help='Dropout for classifier-free guidance (default: 0.25)')
    parser.add_argument('--ema-decay', type=float, default=0.995,
                       help='EMA decay rate (default: 0.995)')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                       help='Use mixed precision training (default: True)')
    parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false',
                       help='Disable mixed precision training')
    
    # Logging and checkpointing
    parser.add_argument('--log-every', type=int, default=5,
                       help='Log training metrics every N epochs (default: 5)')
    parser.add_argument('--checkpoint-every', type=int, default=25,
                       help='Save checkpoint every N epochs (default: 25)')
    parser.add_argument('--model-name', type=str, default='demo_model',
                       help='Name for saved model checkpoints (default: demo_model)')
    
    # Dataloader settings
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--refresh-dataloader', action='store_true', default=True,
                       help='Enable dataloader refresh (default: True)')
    parser.add_argument('--no-refresh-dataloader', dest='refresh_dataloader', action='store_false',
                       help='Disable dataloader refresh')
    parser.add_argument('--refresh-dataloader-every', type=int, default=10,
                       help='Refresh dataloader every N epochs (default: 10)')
    parser.add_argument('--refresh-number', type=int, default=200,
                       help='Number of samples to refresh in dataloader (default: 200)')
    
    # Model architecture
    parser.add_argument('--model-dim', type=int, default=128,
                       help='Model dimension (default: 128)')
    parser.add_argument('--dim-mults', type=str, default='1,2',
                       help='Dimension multipliers as comma-separated list (default: 1,2)')
    
    # Scheduler settings
    parser.add_argument('--scheduler-flat-until', type=float, default=0.4,
                       help='Fraction of epochs to keep learning rate flat (default: 0.4)')
    parser.add_argument('--warmup-steps', type=int, default=100,
                       help='Number of warmup steps for scheduler (default: 100)')
    
    # Logging
    parser.add_argument('--log-file', type=str, default=None,
                       help='Path to log file (default: results/train_<model_name>_<timestamp>.log)')
    
    return parser.parse_args()


def main(args=None):
    """
    Main training function that replicates the Jupyter notebook training loop.
    """
    if args is None:
        args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder = load_autoencoder(half_precision=True)

    ## Parse dim_mults from string to tuple
    dim_mults = tuple(int(x) for x in args.dim_mults.split(','))

    ## Diffusion hyperparameters
    num_epochs = args.num_epochs
    total_timesteps = args.total_timesteps
    learning_rate = args.learning_rate
    beta_t = cosine_beta_schedule(total_timesteps) # Variance Schedule, cosine or linear
    cfg_dropout = args.cfg_dropout
    ema_decay = args.ema_decay
    mixed_precision = args.mixed_precision

    ## Checkpoint
    log_every = args.log_every
    checkpoint_every = args.checkpoint_every
    model_name = args.model_name

    ## Setup logging
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"results/train_{model_name}_{timestamp}.log"
    else:
        log_file = args.log_file
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Open log file for writing
    log_fp = open(log_file, 'w')
    
    def log_print(message):
        """Print to both console and log file"""
        print(message)
        log_fp.write(message + '\n')
        log_fp.flush()  # Ensure it's written immediately
    
    ## Dataloader augmentation refreshing
    refresh_dataloader = args.refresh_dataloader
    refresh_dataloader_every = args.refresh_dataloader_every
    refresh_number = args.refresh_number
    batch_size = args.batch_size
    
    log_print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Model name: {model_name}")
    log_print(f"Hyperparameters:")
    log_print(f"  Learning rate: {learning_rate}")
    log_print(f"  Num epochs: {num_epochs}")
    log_print(f"  Total timesteps: {total_timesteps}")
    log_print(f"  CFG dropout: {cfg_dropout}")
    log_print(f"  EMA decay: {ema_decay}")
    log_print(f"  Mixed precision: {mixed_precision}")
    log_print(f"  Batch size: {batch_size}")
    log_print(f"  Model dim: {args.model_dim}")
    log_print(f"  Dim mults: {dim_mults}")
    log_print(f"  Log every: {log_every} epochs")
    log_print(f"  Checkpoint every: {checkpoint_every} epochs")
    log_print(f"Log file: {log_file}")
    log_print("-" * 80)

    # Create the Unet model
    model = ConditionalUNet(
        dim=args.model_dim, 
        dim_mults=dim_mults, 
        beta_t=beta_t, 
        timesteps=total_timesteps
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = flat_then_decay_w_warmup(
        optimizer=optimizer, 
        num_epochs=num_epochs, 
        flat_until=args.scheduler_flat_until, 
        warmup_steps=args.warmup_steps
    )
    scaler = GradScaler()

    ## Load the training images
    train_imageDataset = load_imageDataset()
    log_print(f"Loaded {len(train_imageDataset)} train_images.")

    epoch_start, ema_params, losses_total, train_loader = read_checkpoint(
        model=model, 
        optimizer=optimizer, 
        scaler=scaler, 
        scheduler=scheduler,
        batch_size=batch_size,
        checkpoint_path=f"model_checkpoints/{model_name}.pth"
    )
    log_print(f"Resuming at epoch: {epoch_start}")

    # Training Loop
    model.train()
    clean()
    
    for epoch in tqdm(range(epoch_start+1, num_epochs)):
        
        train_losses = []
        for i, (batch_conditions, batch_targets, batch_classes) in enumerate(train_loader):

            batch_conditions = batch_conditions.half().to(device)
            batch_targets = batch_targets.half().to(device)
            #batch_classes = batch_classes.to(device)
            
            e = torch.randn_like(batch_targets)
            t = sample_t(bad_data=True, TOTAL_TIMESTEPS=total_timesteps, batch_targets=batch_targets, batch_classes=batch_classes)
            x_t, v_t = forward_sample(x_0=batch_targets, t=t, e=e, alphas_cumprod=model.alphas_cumprod) # Calculates velocity

            # set to null token for unconditional model
            rand_ind = torch.randperm(batch_conditions.shape[0])[:int(cfg_dropout * batch_conditions.shape[0])]
            batch_conditions[rand_ind] = model.class_tokens.to(dtype=batch_conditions.dtype)

            optimizer.zero_grad(set_to_none=True)

            if mixed_precision:
                with torch.amp.autocast(device_type=autocast_device):
                    v_pred = model(x_t, batch_conditions, t)
                    loss = torch.nn.functional.mse_loss(v_t, v_pred)
                 
                if torch.isnan(loss): # If crash   
                    optimizer.zero_grad(set_to_none=True)
                    continue
                    
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            else:
                v_pred = model(x_t, batch_conditions, t)
                loss = F.mse_loss(v_t, v_pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if epoch > 10:
                update_ema(model=model, ema_params=ema_params, ema_decay=ema_decay)
            train_losses.append(loss.item())  # Store the original loss

        if scheduler:
            scheduler.step()
        losses_total.append(np.mean(train_losses))
        
        ## User Logs
        if (epoch+1) % log_every == 0:
            avg_loss = np.mean(train_losses)
            learning_rates = [param_group['lr'] for param_group in optimizer.param_groups]
            current_lr = learning_rates[-1] if learning_rates else 0.0
            log_print(f"Epoch: {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.6f} | LR: {current_lr:.6e}")
            print_training_logs(epoch=epoch, losses_total=losses_total, optimizer=optimizer)

        ## Write Checkpoint
        if (epoch+1) % checkpoint_every == 0 and epoch != 0:
            write_checkpoint(
                epoch=epoch, 
                model=model, 
                optimizer=optimizer, 
                scaler=scaler, 
                ema_params=ema_params, 
                losses_total=losses_total, 
                train_loader=train_loader,
                scheduler=scheduler,
                model_name=model_name
            )

        if refresh_dataloader and (epoch + 1) % refresh_dataloader_every == 0 and epoch != 0:
            train_loader = update_dataloader(
                train_loader=train_loader, 
                train_imageDataset=train_imageDataset, 
                autoencoder=autoencoder, 
                N=refresh_number
            )

        if (epoch+1) % 100 == 0 and epoch != 0:
            clean()
    
    # Close log file
    log_print(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_fp.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)

