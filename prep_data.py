from helpers.read_data_tools import read_data, imageDataset, build_latentDataset, save_latentDataset
from helpers.pre_trained_autoencoder import load_autoencoder
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--H", type=int, default=512, help="Height to resize images")
    parser.add_argument("--W", type=int, default=768, help="Width to resize images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    return parser.parse_args()

## Read data and store autoencoder latents to train the diffusion model once
def main(args=None):

    if args is None:
        args = parse_args()

    train_images, train_labels = read_data(H = int(args.H), W = int(args.W), no_split = True)
    autoencoder = load_autoencoder(half_precision = True)

    train_imageDataset = imageDataset(train_images, train_labels)
    train_latentDataset = build_latentDataset(ImageDataset = train_imageDataset, autoencoder = autoencoder, batch_size = args.batch_size)
    save_latentDataset(train_latentDataset, save_name = "train_latentDataset", save_path = "./images/data_checkpoints/")


if __name__ == "__main__":
    args = parse_args()
    main(args)