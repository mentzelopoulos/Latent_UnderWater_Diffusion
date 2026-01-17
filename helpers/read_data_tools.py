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
from torchvision import transforms
import torchvision.transforms.functional as FF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import random
import gzip

from helpers.artificial_corruption_utils import corrupt_fast
from helpers.pre_trained_autoencoder import encode


device = "cuda" if torch.cuda.is_available() else "cpu"
autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

## Read data from folder
@torch.no_grad()
def read_data(root_dir: str = "images", H: int = 512, W: int = 768, seed: int = 42, test_split: float = 0.1, no_split = False):
    """
    Reads images from root_dir, splits into train/test, and returns tensors + labels.

    Args:
        root_dir (str): path to image directory (subfolders = classes)
        H (int): output image height [Standard: 512]
        W (int): output image width [Standard: 768]
        seed (int): random seed for reproducibility
        test_split (float): fraction of images to use for test set [Standard: 0.1 for 10% test set]
        no_split (bool): whether to split data into train/test [Standard: False]
        
    Returns:
        train_images (list[torch.Tensor]): C x H x W images
        train_labels (list[int]): class labels
        test_images (list[torch.Tensor]): C x H x W images
        test_labels (list[int]): class labels
    """

    transform = transforms.Compose([
        transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    ## No split used for test or unconditional
    if no_split:
        
        classes = sorted([
            folder for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder)) 
               and not folder.startswith('.') 
               and folder != '.ipynb_checkpoints'
        ])
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
    
        train_images, train_labels = [], []
        skipped = []
    
        for class_name in classes:
            
            class_path = os.path.join(root_dir, class_name)
            image_paths = sorted([
                os.path.join(class_path, fname)
                for fname in os.listdir(class_path)
                if fname.lower().endswith(('.jpg', '.jpeg'))
            ])
            if not image_paths:
                print(f"No images found in folder {class_name}" )
                continue
            for path in tqdm(image_paths, desc=f"Loading {class_name}"):
                try:
                    img = Image.open(path).convert("RGB")
                    img = transform(img)

                    # Ensure correct orientation if needed
                    if img.shape[-1] < img.shape[-2]:  # W < H
                        img = torch.transpose(img, 1, 2)

                    train_images.append(img)
                    train_labels.append(class_to_idx[class_name])
                except Exception as e:
                    skipped.append((path, str(e)))
        return train_images, train_labels
        
    else: ## Train/test split
        generator = torch.Generator().manual_seed(seed)    
    
        classes = sorted([
            folder for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder)) 
               and not folder.startswith('.') 
               and folder != '.ipynb_checkpoints'
        ])
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
    
        train_images, train_labels = [], []
        test_images, test_labels = [], []
        skipped = []
    
        for class_name in classes:
            class_path = os.path.join(root_dir, class_name)
            image_paths = sorted([
                os.path.join(class_path, fname)
                for fname in os.listdir(class_path)
                if fname.lower().endswith(('.jpg', '.jpeg'))
            ])
            if not image_paths:
                continue
            
            # Shuffle images
            idx = torch.randperm(len(image_paths), generator=generator).tolist()
            image_paths = [image_paths[i] for i in idx]
    
            n_test = max(1, int(test_split * len(image_paths)))
            test_paths, train_paths = image_paths[:n_test], image_paths[n_test:]
    
            for split_paths, split_imgs, split_labels, split_name in [
                (train_paths, train_images, train_labels, "train"),
                (test_paths, test_images, test_labels, "test"),
            ]:
                for path in tqdm(split_paths, desc=f"Loading {split_name} {class_name}"):
                    try:
                        img = Image.open(path).convert("RGB")
                        img = transform(img)
    
                        # Ensure correct orientation if needed
                        if img.shape[-1] < img.shape[-2]:  # W < H
                            img = torch.transpose(img, 1, 2)
    
                        split_imgs.append(img)
                        split_labels.append(class_to_idx[class_name])
                    except Exception as e:
                        skipped.append((path, str(e)))
    
        if skipped:
            print(f"Skipped {len(skipped)} files.")

        return train_images, train_labels, test_images, test_labels

    



## Latent Dataset
class latentDataset(Dataset):
    def __init__(self, conditions: torch.Tensor, targets: torch.Tensor, classes: torch.Tensor, raw_latents: torch.Tensor = None):
        super().__init__()
        if raw_latents is not None:
            self.raw_latents = raw_latents
        self.conditions = conditions
        self.targets = targets
        self.classes = classes

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, idx):
        return self.conditions[idx], self.targets[idx], self.classes[idx]



def save_latentDataset(
    LatentDataset: latentDataset,
    save_name: str,
    save_path: str = "./images/data_checkpoints/",
    half_precision: bool = True,
    compressed: bool = False
):
    data = {
        "conditions": LatentDataset.conditions.half() if half_precision else LatentDataset.conditions.float(),
        "targets": LatentDataset.targets.half() if half_precision else LatentDataset.targets.float(),
        "classes": LatentDataset.classes,
    }
    if hasattr(LatentDataset, "raw_latents") and LatentDataset.raw_latents is not None:
        data["raw_latents"] = LatentDataset.raw_latents

    os.makedirs(save_path, exist_ok=True)

    file = os.path.join(save_path, save_name + ".pt")
    if compressed:
        file += ".gz"
        with gzip.open(file, "wb") as f:
            torch.save(data, f)
    else:
        torch.save(data, file)


def load_latentDataset(save_name: str = "train_latentDataset", save_path: str = "./images/data_checkpoints/") -> latentDataset:
    pt_file = os.path.join(save_path, save_name + ".pt")
    gz_file = pt_file + ".gz"

    if os.path.isfile(gz_file):
        # Load gzip compressed file
        with gzip.open(gz_file, "rb") as f:
            data = torch.load(f)
    elif os.path.isfile(pt_file):
        # Load regular .pt file
        data = torch.load(pt_file)
    else:
        raise FileNotFoundError(f"Neither {pt_file} nor {gz_file} exists.")
        
    print(f"Loaded data as type {data['conditions'].dtype}") 
    # Create latentDataset object
    return latentDataset(
        conditions=data["conditions"],
        targets=data["targets"],
        classes=data["classes"],
        raw_latents=data.get("raw_latents", None)  # handle optional raw_latents
    )


    
## Image Dataset
class imageDataset(Dataset):
    def __init__(self, images: list[torch.Tensor], classes: torch.Tensor):
        super().__init__()
        self.images = images
        self.classes = classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.classes[idx]


def save_imageDataset(
    dataset: imageDataset,
    save_name: str,
    save_path: str = "./images/data_checkpoints/",
    batch_size: int = 512  # tune based on GPU/CPU RAM
):
    os.makedirs(save_path, exist_ok=True)

    n = len(dataset)
    images_uint8_batches = []
    
    # process in chunks
    for i in range(0, n, batch_size):
        batch = torch.stack(dataset.images[i:i+batch_size])  # (B, C, H, W)
        batch = batch.mul(255).clamp(0, 255).to(torch.uint8)
        images_uint8_batches.append(batch)

    # concatenate all chunks
    images_uint8 = torch.cat(images_uint8_batches, dim=0)  # (N, C, H, W)

    data = {
        "images": images_uint8,
        "classes": dataset.classes,
    }

    torch.save(data, os.path.join(save_path, save_name + ".pt"))


def load_imageDataset(save_name="train_imageDataset", save_path="./images/data_checkpoints/"):
    data = torch.load(os.path.join(save_path, save_name + ".pt"), map_location="cpu")

    # back to float32 [0,1] range
    images = data["images"].to(torch.float32).div(255)

    # you can either return list or keep as tensor
    return imageDataset(images=list(images), classes=data["classes"])


'''
def save_imageDataset(
    ImageDataset: imageDataset,
    save_name: str,
    save_path: str = "./images/data_checkpoints/",
    half_precision: bool = True,
    compressed: bool = False
):
    data = {
        "images": ImageDataset.images,
        "classes": ImageDataset.classes,
    }

    os.makedirs(save_path, exist_ok=True)

    file = os.path.join(save_path, save_name + ".pt")
    if compressed:
        file += ".gz"
        with gzip.open(file, "wb") as f:
            torch.save(data, f)
    else:
        torch.save(data, file)

def load_imageDataset(save_name="train_imageDataset", save_path: str = "./images/data_checkpoints/"):
    pt_file = os.path.join(save_path, save_name + ".pt")
    gz_file = pt_file + ".gz"

    if os.path.isfile(gz_file):
        # Load gzip compressed file
        with gzip.open(gz_file, "rb") as f:
            data = torch.load(f)
    elif os.path.isfile(pt_file):
        # Load regular .pt file
        data = torch.load(pt_file)
    else:
        raise FileNotFoundError(f"Neither {pt_file} nor {gz_file} exists.")

    return imageDataset(images=data["images"], classes=data["classes"])
'''

@torch.no_grad()
def augment_batch(batch: torch.Tensor, sharpness_transform=transforms.RandomAdjustSharpness(0.5, p=1.0)) -> torch.Tensor:
    """
    Simple batch augmentations: horizontal flip and optional sharpness.
    Input:  BxCxHxW tensor
    Output: augmented BxCxHxW tensor
    """
    assert batch.ndim == 4
    batch = batch.to(device)

    if random.random() > 0.5: batch = FF.hflip(batch)
    # if random.random() > 0.5: batch = FF.vflip(batch)  # optional
    if random.random() > 0.5: batch = sharpness_transform(batch)
    
    return batch.cpu()

@torch.no_grad()
def beautify_batch(batch: torch.Tensor) -> torch.Tensor:
    """
    Enhance batch visually: random contrast, saturation, hue, autocontrast, and sharpness.
    Input:  BxCxHxW tensor
    Output: beautified BxCxHxW tensor
    """
    batch = batch.to(device)

    if random.random() > 0.35:
        batch = FF.adjust_contrast(batch, 1 + torch.rand(1, device=device) * 0.25)
        batch = FF.adjust_saturation(batch, 1 + torch.rand(1, device=device) * 0.25)
        batch = FF.adjust_hue(batch, random.choice([torch.rand(1, device=device)*0.01, -torch.rand(1, device=device)*0.01]))
    else:
        batch = FF.autocontrast(batch)

    batch = FF.adjust_sharpness(batch, 1.25 + torch.rand(1, device=device) * 1.5)
    return batch.cpu()

@torch.no_grad()
@torch.amp.autocast(device_type=autocast_device)
def build_latentDataset(
    ImageDataset: imageDataset,              
    autoencoder,
    beautify: bool = True,      
    batch_size: int = 32
):
    if len(ImageDataset) == 0:
        raise ValueError("Cannot build latentDataset from empty imageDataset")

    N_images = len(ImageDataset)
    sample_image, _ = ImageDataset[0]
    C, H, W = sample_image.shape

    raw_latents = torch.zeros(N_images, 4, H//8, W//8)
    conditions = torch.zeros_like(raw_latents)
    targets = torch.zeros_like(raw_latents)
    all_classes = torch.zeros(N_images, dtype=torch.long)

    loader = DataLoader(ImageDataset, batch_size=batch_size, shuffle=False)

    for i, (images, classes) in enumerate(tqdm(loader, desc = "Batch encoding for training dataset")):
        images = images.to(device)
        classes = classes.to(device)

        aug_images = augment_batch(images)  # Apply augmentation
        corrupted_images = corrupt_fast(aug_images, apply_inpainting=True)
        beautified_images = beautify_batch(aug_images) if beautify else aug_images

        start = i * batch_size
        end = start + images.size(0)

        raw_latents[start:end] = encode(images, autoencoder)
        conditions[start:end] = encode(corrupted_images, autoencoder)
        targets[start:end] = encode(beautified_images, autoencoder)
        all_classes[start:end] = classes

    return latentDataset(
        conditions=conditions, 
        targets=targets, 
        classes=all_classes, 
        raw_latents=raw_latents
    )

        
            



