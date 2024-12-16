import os

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from UNet import UNet
from gauss import Gauss


def load_checkpoint(path, model, optimizer=None):
    """
    Load the model and optimizer states from a checkpoint.
    Args:
        path: Path to the checkpoint.
        model: PyTorch model to load into.
        optimizer: (Optional) PyTorch optimizer to load into.
    Returns:
        The epoch at which training was saved.
    """
    if not os.path.exists(path):
        print(f"Checkpoint file '{path}' does not exist. Skipping load.")
        return
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path}, epoch: {epoch}")
    return epoch


def save_checkpoint(model, optimizer, epoch, path="unet_checkpoint.pth"):
    """
    Save the model and optimizer states.
    Args:
        model: PyTorch model to save.
        optimizer: PyTorch optimizer to save.
        epoch: Current epoch number.
        path: Path to save the checkpoint.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, path)
    print(f"Checkpoint saved at {path}")


# Assuming `model` is an instance of your UNet
def denoise(model, noisy_images, timesteps, device):
    """
    Denoise a batch of images.
    Args:
        model: The UNet model.
        noisy_images: Tensor (B, C, H, W), noisy input images.
        timesteps: Tensor (B,), conditioning timestep values.
        device: Torch device to run the model on.
    Returns:
        Denoised images.
    """
    noisy_images = noisy_images.to(device)
    timesteps = timesteps.to(device)
    with torch.no_grad():
        denoised_images = model(noisy_images, timesteps)
    return denoised_images


def main():
    # Get device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform for dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
    ])

    # Load dataset (e.g., CIFAR-10)
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    # Model , optimize , loss
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    n_epochs = 30
    timesteps = 1000  # Standard deviation of added noise
    start_epoch = load_checkpoint("unet.pth", model, optimizer)

    # Noise
    gauss = Gauss(T=1000, beta_start=0.0001, beta_end=0.02,device=device)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch")
        for batch in progress_bar:
            images, _ = batch  # ignore labels

            # Timesteps (can vary depending on the application, e.g., diffusion timesteps)
            t = torch.randint(0, timesteps, (images.size(0),), device=device)
            t.to(device)

            noised_images, noise = gauss.q_sample(images, t)
            noise.to(device)

            # Forward pass
            predicted_noise = model(noised_images, t)

            # Compute loss
            loss = criterion(predicted_noise, noise)
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
        save_checkpoint(model, optimizer, epoch + 1, path=f"unet.pth")
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")


if __name__ == '__main__':
    main()
