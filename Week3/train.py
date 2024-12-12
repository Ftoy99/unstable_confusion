import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
# This next 2 imports are needed 4 some reason
import safetensors
from safetensors.torch import load_model, save_model, save_file
from diffusers import AutoencoderKL
from Week3.ldm2 import LatentDiffusionModel


def save_model(model, file_path):
    """
    Save the model's state dictionary to a file.

    Args:
        model: The model to save.
        file_path: The file path where the model will be saved.
    """
    state_dict = model.state_dict()
    save_file(state_dict, file_path)
    print(f"Model saved to {file_path}")


def train_ldm_model(ldm, vae, dataloader, epochs, lr, device):
    # Optimizer and loss function
    optimizer = torch.optim.Adam(ldm.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()

    ldm.train()
    vae.eval()

    for epoch in range(epochs):
        for batch_idx, images in enumerate(dataloader):
            images = images[0]
            images = images.to(device)

            # Encode images into latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()

            # Add noise to the latents (forward diffusion)
            t = torch.randint(0, 1000, (latents.size(0),), device=device).long()
            x_t, noise = ldm(latents, t)

            # Predict noise
            optimizer.zero_grad()
            predicted_noise = ldm.unet(x_t, t)

            # Calculate loss
            loss = mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        save_model(ldm, "weights/ldm.safetensors")
    print("Training complete!")


if __name__ == "__main__":
    # Initialize model and training parameters
    ldm = LatentDiffusionModel()  # Replace with your LDM model initialization

    # Autoencoder
    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    vae = AutoencoderKL.from_single_file(url)

    # CIFAR-10 dataset setup
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Training parameters
    epochs = 10
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to device
    ldm.to(device)
    vae.to(device)

    train_ldm_model(ldm, vae, dataloader, epochs, lr, device)
