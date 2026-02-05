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


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import os


from helpers.read_data_tools import imageDataset, latentDataset, read_data
from helpers.pre_trained_autoencoder import decode
from helpers.plotting_utils import plot_image
from helpers.diffusion_utils_conditional import sample



def load_model_and_test_loader(
    model: torch.nn.Module,
    batch_size: int = 16,
    dataset_path: str = 'images/data_checkpoints/test_imageDataset.pt',
    checkpoint_path: str = "model_checkpoints/demo_model.pth",
    strict = False
    ):
    
    ## Load ema model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "ema_params" in checkpoint:
        model.load_state_dict(checkpoint['ema_params'], strict=strict)
        print(f"Model updated with EMA parameters, loaded with strict = {strict}")
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"Warning: EMA parameters not found, but loaded regular parameters, loaded woth strict = {strict}")
    
    try:
        data = torch.load(dataset_path, map_location="cpu")
        dataset = imageDataset(
            images = data["images"],
            classes = data["classes"]
        )
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return test_loader
    except:
        print("Warning: data checkpoint does not exist")
    
    return 

@torch.no_grad()
@torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
def generate_unconditional(
    model: nn.Module,
    autoencoder: nn.Module,
    num_samples: int = 16,
    total_timesteps:int = 1000,
    quick: bool = False,
    quick_inference_steps: int = 50,
    show_images: bool = True,
    save_images_to_results: bool = True,
    save_image_path: str = "./results/generated_images/",
    find_nearest_neighbor: bool = False,
    train_imageDataset: imageDataset = None,
):

    if save_images_to_results == False and show_images == False:
        print("Both saving and showing images are disabled. Showing but not saving by default")
        show_images = True

    if find_nearest_neighbor and train_loader is None:
        print("Warning: Cannot find closet samples, training set dataloader not passed")
        find_nearest_neighbor = False

    if not os.path.isdir(save_image_path):
        os.makedirs(save_image_path, exist_ok=True)

    ## Sample model unconditionally
    samples = sample(
        model = model, 
        autoencoder = autoencoder, 
        corrupted_img = None, 
        num_samples = num_samples,
        total_timesteps = total_timesteps,
        quick = quick, 
        num_timesteps = quick_inference_steps, 
        cfg_guidance = False, 
        cfg_scale = 2)

    ## Keep clean images only
    samples = samples[-1]
    
    for i in range(samples.shape[0]):

        curr_sample = samples[[i]]
        curr_sample = decode(z = curr_sample, autoencoder = autoencoder).float()
        curr_sample = curr_sample[0].detach()

        if show_images:
            plot_image(x = curr_sample, title = f"LOBSTgER Generated Sample ({i+1}/{samples.shape[0]})")

        if save_images_to_results:
           save_image_to_folder(image = curr_sample, save_image_path = save_image_path, is_nn = False, index = i)

        if find_nearest_neighbor:
            closest_sample = find_closest_sample(image = curr_sample, train_imageDataset = train_imageDataset)
            if show_images:
                plot_image(x = closest_sample, title = f"Nearest neighbor for image  ({i} / {samples.shape[0]}")
            if save_images_to_results:
                save_image_to_folder(save_image_path = save_image_path, image = curr_sample, is_nn = True, index = i)



@torch.no_grad()
@torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
def generate_conditional(
    model: nn.Module,
    autoencoder: nn.Module,
    corrupted_img: torch.Tensor = None,
    total_timesteps:int = 1000, 
    quick: bool = False,
    quick_inference_steps: int = 50,
    cfg:bool = False,
    cfg_scale: float = 2.5,
    show_images: bool = True,
    save_images_to_results: bool = True,
    save_image_path: str = "./results/generated_images/",
    find_nearest_neighbor: bool = False,
    train_imageDataset: imageDataset = None,
):

    if corrupted_img is None:
        assert False, "No corrupted images were passed"

    if save_images_to_results == False and show_images == False:
        print("Both saving and showing images are disabled. Showing but not saving by default")
        show_images = True

    if find_nearest_neighbor and train_loader is None:
        print("Warning: Cannot find closet samples, training set dataloader not passed")
        find_nearest_neighbor = False

    if not os.path.isdir(save_image_path):
        os.makedirs(save_image_path, exist_ok=True)

    ## Sample model unconditionally
    samples = sample(
        model = model, 
        total_timesteps = total_timesteps,
        autoencoder = autoencoder, 
        corrupted_img = corrupted_img, 
        quick = quick, 
        num_timesteps = quick_inference_steps, 
        cfg_guidance = cfg, 
        cfg_scale = cfg_scale)

    ## Keep clean images only
    samples = samples[-1]
    
    for i in range(samples.shape[0]):

        curr_sample = samples[[i]]
        curr_sample = decode(z = curr_sample, autoencoder = autoencoder).float()
        curr_sample = curr_sample[0].detach()

        if show_images:
            plot_image(x = corrupted_img[[i]], title = f"Condition ({i+1}/{samples.shape[0]})")
            plot_image(x = curr_sample, title = f"LOBSTgER Generated Sample ({i+1}/{samples.shape[0]})")
            
        if save_images_to_results:
           save_image_to_folder(image = curr_sample, save_image_path = save_image_path, is_nn = False, index = i)

        if find_nearest_neighbor:
            closest_sample = find_closest_sample(image = curr_sample, train_imageDataset = train_imageDataset)
            if show_images:
                plot_image(x = closest_sample, title = f"Nearest neighbor for image  ({i} / {samples.shape[0]}")
            if save_images_to_results:
                save_image_to_folder(save_image_path = save_image_path, image = curr_sample, is_nn = True, index = i)


############################### Helpers

@torch.no_grad()
def save_image_to_folder(image: torch.Tensor, save_image_path = "./results/generated_images/", is_nn: bool = False, index: int = None):
    tosave_img = image.squeeze().detach().cpu().numpy().transpose(1,2,0)
    tosave_img = (tosave_img*255).astype('uint8')
    tosave_img = Image.fromarray(tosave_img)
    str_index = str(index) if index is not None else ""
    save_name = save_image_path + 'unconditional_' + str_index +'.png' if (not is_nn) else save_image_path + "NN" + 'unconditional_' + str_index +'.png'
    tosave_img.save(save_name)

@torch.no_grad()
def find_closest_sample(image: torch.Tensor, train_imageDataset: imageDataset):

    assert image.ndim == 3 or (image.ndim == 4 and image.shape[0] == 1), "Please provide a single image"
    assert image.shape[0] == 3 or image.shape[1] == 3, "Please provide 3 channel image in pixel space"

    image.squeeze_() 
    assert image.ndim == 3

    min_dist = float("inf")
    for i, img in enumerate(train_imageDataset.images):
        dist = torch.sqrt(torch.sum((image.cpu() - img) ** 2))
        if dist < min_dist:
            min_dist = dist
            indx = i
            
    return train_imageDataset.images[indx]
        

    


    