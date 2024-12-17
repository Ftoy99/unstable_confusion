import os

import torch
from matplotlib import pyplot as plt
from torch import optim

from UNet import UNet


def load_checkpoint(path, model, optimizer=None):
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


def denoise(model, noisy_images, timesteps):
    # Initialize the noisy image as the starting point for denoising
    # Iterate through the timesteps in reverse (from T-1 to 0)
    for t in reversed(range(timesteps)):
        # Create a tensor of shape [batch_size] filled with the current timestep
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise = model(noisy_images, t_tensor)
        print(f"Step {t}")
        noisy_images = noisy_images-noise
    return noisy_images


batch_size = 1  # We want to generate one image
img_channels = 3  # For RGB images
height, width = 32, 32  # Example dimensions
timesteps = 1000
# Example of running the denoising process
x = torch.randn(batch_size, img_channels, height, width)  # Noisy input image
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the UNet model
unet = UNet().to(device)
optimizer = optim.Adam(unet.parameters(), lr=1e-4)

# Load the checkpoint
load_checkpoint("unet.pth", unet, optimizer)

# Denoising over 1000 timesteps
output = denoise(unet, x.to(device), 1000)

# Convert the output tensor to a valid image for visualization
output_image = output.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())  # Normalize to [0, 1]
output_image = output_image.transpose(1, 2, 0)  # Convert to HxWxC format for visualization

# Plot the denoised image
plt.imshow(output_image)
plt.axis('off')
plt.show()
