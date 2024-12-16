import os

import torch
from matplotlib import pyplot as plt
from torch import optim

from UNet import UNet


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


def denoise(model, noisy_images, timesteps, device, num_timesteps=1000):
    # Initialize the noisy image as the starting point for denoising
    denoised_images = noisy_images

    # Iterate through the timesteps in reverse (from T-1 to 0)
    for t in reversed(range(num_timesteps)):
        # Generate the timestep tensor for this specific timestep
        timestep = torch.tensor([t] * noisy_images.size(0), device=device)

        # Denoise the image at this timestep
        denoised_images = model(denoised_images, timesteps[t])

        # You can also add noise here depending on your specific reverse process setup
        # If your model does not include the noise schedule internally, you can add it here manually
        # Example: denoised_images = denoised_images + noise_at_timestep

    return denoised_images


batch_size = 1  # We want to generate one image
img_channels = 3  # For RGB images
height, width = 32, 32  # Example dimensions

# Example of running the denoising process
x = torch.randn(batch_size, img_channels, height, width)  # Noisy input image
t = torch.randint(0, 1000, (batch_size,))  # Random timesteps (just for the example)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the UNet model
unet = UNet().to(device)
optimizer = optim.Adam(unet.parameters(), lr=1e-4)

# Load the checkpoint
load_checkpoint("unet.pth", unet, optimizer)

# Denoising over 1000 timesteps
output = denoise(unet, x.to(device), t.to(device), device)

# Convert the output tensor to a valid image for visualization
output_image = output.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())  # Normalize to [0, 1]
output_image = output_image.transpose(1, 2, 0)  # Convert to HxWxC format for visualization

# Plot the denoised image
plt.imshow(output_image)
plt.axis('off')
plt.show()
