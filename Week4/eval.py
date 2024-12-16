import os

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.metrics import structural_similarity as ssim
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from UNet import UNet


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


# Add synthetic noise during training
def prepare_batch(batch, noise_level):
    images, _ = batch  # We don't need labels for denoising
    noisy_images = add_noise(images, noise_level)
    return noisy_images, images


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


def calculate_psnr(clean_image_np, denoised_image_np, data_range=1.0):
    # Compute the Mean Squared Error (MSE)
    mse = np.mean((clean_image_np - denoised_image_np) ** 2)
    if mse == 0:
        return 100  # If images are identical, return very high PSNR
    return 10 * np.log10((data_range ** 2) / mse)


def evaluate_denoising(model, dataloader, noise_level, device):
    model.eval()
    psnr_sum, ssim_sum, n_samples = 0, 0, 0
    total_images = 0
    with torch.no_grad():
        for batch in dataloader:
            noisy_images, clean_images = prepare_batch(batch, noise_level)
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

            # Generate timesteps for the denoising process
            timesteps = torch.tensor([noise_level] * noisy_images.size(0), device=device)

            # Denoise the images
            denoised_images = model(noisy_images, timesteps).cpu()

            # Evaluate SSIM and PSNR
            for i in range(clean_images.size(0)):
                clean_image_np = clean_images[i].cpu().numpy().transpose(1, 2, 0)  # HWC
                denoised_image_np = denoised_images[i].cpu().numpy().transpose(1, 2, 0)  # HWC

                # Ensure the window size is appropriate for the image dimensions
                win_size = min(denoised_image_np.shape[0], denoised_image_np.shape[1], 7)
                if win_size % 2 == 0:
                    win_size -= 1  # Ensure win_size is odd

                # SSIM calculation
                ssim_sum += ssim(
                    clean_image_np,
                    denoised_image_np,
                    multichannel=True,
                    win_size=win_size,  # Explicitly set the window size
                    channel_axis=-1,  # Specify the channel axis for multichannel images
                    data_range=1.0,
                )

                # PSNR calculation
                psnr_sum += calculate_psnr(clean_image_np, denoised_image_np, data_range=1.0)

                total_images += 1

    avg_psnr = psnr_sum / total_images
    avg_ssim = ssim_sum / total_images
    print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.2f}")
    return avg_psnr, avg_ssim


if __name__ == '__main__':
    # Transform for dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset (e.g., CIFAR-10)
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    noise_level = 0.1  # Standard deviation of added noise
    unet = UNet().to(device)
    optimizer = optim.Adam(unet.parameters(), lr=1e-4)
    load_checkpoint("unet.pth", unet, optimizer)

    evaluate_denoising(unet, dataloader, noise_level, device)
