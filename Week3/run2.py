import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from matplotlib import pyplot as plt

# This next 2 imports are needed 4 some reason
import safetensors
from safetensors.torch import load_model, save_model

from ldm2 import LatentDiffusionModel

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using device:", device)
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using device:", device)

def display_image(image_title_pairs: tuple[Image, str]):
    # Determine the number of images
    num_images = len(image_title_pairs)

    # Create a figure with subplots (one row, num_images columns)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    # If there's only one image, axes won't be an array, so we handle that
    if num_images == 1:
        axes = [axes]

    # Iterate over the list of image-title pairs and plot them
    for ax, (img, title) in zip(axes, image_title_pairs):
        ax.imshow(img)
        ax.axis('off')  # Turn off axis
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def load_image(path) -> Image:
    return Image.open(path)


def run():
    ldm = LatentDiffusionModel(device=device).to(device)

    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    model = AutoencoderKL.from_single_file(url).to(device)

    ldm.eval()
    model.eval()

    image = load_image("img/cat.jpg")
    image = image.resize((128, 128))

    # Convert the image to a tensor
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

    # Encode the image
    with torch.no_grad():
        latent_representation = model.encode(image_tensor).latent_dist.sample()

    steps = 200
    timesteps = torch.arange(0, steps)
    # latent_representation = add_noise(latent_representation,t)
    for t in reversed(range(0, timesteps.size(0))):
        print(f"Starting step {t}")
        t_tensor = torch.tensor([t], dtype=torch.long).to(device)
        latent_representation = ldm.reverse_diffusion(latent_representation, t_tensor)

    # Decode the latent representation back to an image
    with torch.no_grad():
        reconstructed_image_tensor = model.decode(latent_representation)['sample']

    # Convert the tensor back to an image
    reconstructed_image_tensor = reconstructed_image_tensor.squeeze(0).permute(1, 2, 0)  # Shape: [H, W, 3]
    reconstructed_image = (reconstructed_image_tensor.clamp(0, 1) * 255).byte().numpy()
    reconstructed_image = Image.fromarray(reconstructed_image)

    images_to_display = [(image, "Resized original"), (reconstructed_image, "Reconstructed Image")]

    display_image(images_to_display)


if __name__ == '__main__':
    run()
