import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


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


def evaluate_denoising(model, dataloader, noise_level, device):
    model.eval()
    psnr_sum, ssim_sum, n_samples = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            noisy_images, clean_images = prepare_batch(batch, noise_level)
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)

            timesteps = torch.zeros(noisy_images.size(0), device=device)  # e.g., constant timestep
            denoised_images = model(noisy_images, timesteps).cpu()

            for i in range(clean_images.size(0)):
                psnr_sum += peak_signal_noise_ratio(clean_images[i].numpy(), denoised_images[i].numpy())
                ssim_sum += structural_similarity(clean_images[i].numpy().transpose(1, 2, 0),
                                                  denoised_images[i].numpy().transpose(1, 2, 0),
                                                  multichannel=True)
                n_samples += 1

    avg_psnr = psnr_sum / n_samples
    avg_ssim = ssim_sum / n_samples
    print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.2f}")
