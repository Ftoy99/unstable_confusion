from datetime import datetime
import os

import torch
from diffusers import AutoencoderKL
from matplotlib import pyplot as plt
from torch import optim
import safetensors.torch
from torchvision.transforms import transforms
from tqdm import tqdm

from UNetLCA import UNet
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
    progress_bar = tqdm(reversed(range(timesteps)), desc=f"Inference ", unit="step")
    for t in progress_bar:
        # Create a tensor of shape [batch_size] filled with the current timestep
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        with torch.no_grad():
            noise = model(noisy_images, t_tensor)

        # Perform the denoising step (reverse diffusion)
        noisy_images = gauss.p_sample(noisy_images, t, noise, True)

        # Convert tensor to PIL image and save
        pil_image = to_pil(noisy_images[0].cpu())  # Convert to PIL and move to CPU
        img_filename = f"{save_dir}/img_{t}_{current_time}.png"
        pil_image.save(img_filename)
    # Return the denoised image
    return noisy_images


batch_size = 1  # We want to generate one image
img_channels = 4  # For RGB images
height, width = 4, 4  # Example dimensions
timesteps = 1000
# Example of running the denoising process
x = torch.randn(batch_size, img_channels, height, width)  # Noisy input image
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the UNet model
unet = UNet(image_channels=4, norm_group=2).to(device)
unet.eval()
optimizer = optim.Adam(unet.parameters(), lr=1e-4)

# Load the checkpoint
load_checkpoint("unetLCA.pth", unet, optimizer)

url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be a local file
ae = AutoencoderKL.from_single_file(url)
ae.to(device)
# Denoising over 1000 timesteps
output = denoise(unet, x.to(device), 1000, 1, device)


# Convert the output tensor to a valid image for visualization
output_image = ae.decode(output)["sample"].squeeze(0).detach().cpu().numpy()  # Remove batch dimension
output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())  # Normalize to [0, 1]
output_image = output_image.transpose(1, 2, 0)  # Convert to HxWxC format for visualization

# Plot the denoised image
plt.imshow(output_image)
plt.axis('off')
plt.show()
