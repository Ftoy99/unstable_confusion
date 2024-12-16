import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from UNet import UNet

# Transform for dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
])

# Load dataset (e.g., CIFAR-10)
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)


def add_noise(images: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Add Gaussian noise to images.
    Args:
        images: Tensor of shape (B, C, H, W), the clean images.
        noise_level: Float, the standard deviation of the noise.
    Returns:
        Noisy images of the same shape as input.
    """
    noise = torch.randn_like(images) * noise_level
    return images + noise


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


# Add synthetic noise during training
def prepare_batch(batch, noise_level):
    images, _ = batch  # We don't need labels for denoising
    noisy_images = add_noise(images, noise_level)
    return noisy_images, images


# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training loop
n_epochs = 1
noise_level = 0.1  # Standard deviation of added noise
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch")
    for batch in progress_bar:
        noisy_images, clean_images = prepare_batch(batch, noise_level)
        noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

        # Timesteps (can vary depending on the application, e.g., diffusion timesteps)
        timesteps = torch.randint(0, 1000, (noisy_images.size(0),), device=device) / 1000.0

        # Forward pass
        denoised_images = model(noisy_images, timesteps)

        # Compute loss
        loss = criterion(denoised_images, clean_images)
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")
