from datetime import datetime
import os

import torch
from matplotlib import pyplot as plt
from torch import optim
from torchvision.transforms import transforms

from UNet import UNet
from gauss import Gauss


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


def denoise(model, noisy_images, timesteps, batch_size, device):
    gauss = Gauss(T=1000, beta_start=0.0001, beta_end=0.02, device=device)

    # Create a directory with the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"results/{current_time}"
    os.makedirs(save_dir, exist_ok=True)

    # Transformation to convert tensor to PIL image
    to_pil = transforms.ToPILImage()

    # Iterate through the timesteps in reverse (from T-1 to 0)
    for t in reversed(range(timesteps)):
        # Create a tensor of shape [batch_size] filled with the current timestep
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Generate model prediction (denoise step)
        noise = model(noisy_images, t_tensor)
        print(f"Step {t}")

        # Convert tensor to PIL image and save
        pil_image = to_pil(noisy_images[0].cpu())  # Convert to PIL and move to CPU
        img_filename = f"{save_dir}/img_{t}_{current_time}.png"
        pil_image.save(img_filename)
        print(f"Saved image at timestep {t} to {img_filename}")

        # Perform the denoising step (reverse diffusion)
        noisy_images = gauss.p_sample(noisy_images, t, noise, True)

    # Return the denoised image
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
output = denoise(unet, x.to(device), 1000, 1, device)

# Convert the output tensor to a valid image for visualization
output_image = output.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())  # Normalize to [0, 1]
output_image = output_image.transpose(1, 2, 0)  # Convert to HxWxC format for visualization

# Plot the denoised image
plt.imshow(output_image)
plt.axis('off')
plt.show()
