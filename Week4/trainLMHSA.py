import os

from diffusers import AutoencoderKL
import torch
import safetensors.torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from UNetLMHSA import UNet
from EMA import EMA
from gauss import Gauss


def view_img(img_tensor):
    # Convert the output tensor to a valid image for visualization
    output_image = img_tensor.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
    output_image = (output_image - output_image.min()) / (
            output_image.max() - output_image.min())  # Normalize to [0, 1]
    output_image = output_image.transpose(1, 2, 0)  # Convert to HxWxC format for visualization

    # Plot the denoised image
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()


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


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, path)
    print(f"Checkpoint saved at {path}")


def main():
    # Get device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform for dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
    ])

    # Load dataset (e.g., CIFAR-10)
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be a local file
    ae = AutoencoderKL.from_single_file(url)

    # Model , optimize , loss
    model = UNet(image_channels=4, norm_group=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Initialize EMA
    ema = EMA(model, beta=0.999)
    # Training loop
    n_epochs = 20
    timesteps = 1000  # Standard deviation of added noise
    load_checkpoint("unetLMHSA.pth", model, optimizer)

    # Noise
    gauss = Gauss(T=1000, beta_start=0.0001, beta_end=0.02, device=device)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch")
        for batch in progress_bar:
            images, _ = batch  # ignore labels
            # view_img(images[0])
            # Timesteps (can vary depending on the application, e.g., diffusion timesteps)
            t = torch.randint(0, timesteps, (images.size(0),), device=device)
            t = t.to(device)

            encoded_images = ae.encode(images)
            encoded_images = encoded_images["latent_dist"].mean
            # ae.decode(encoded_images["latent_dist"].mean)["sample"] wth
            # encoded_images.shape = 2,8,4,4 ? B,C,H,W ?
            noised_images, noise = gauss.q_sample(encoded_images, t)

            # view_img(ae.decoder(noised_images)[0])
            # view_img(gauss.p_sample(noised_images[0], t[0], noise[0]))
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

            # Update EMA weights
            ema.update()

            progress_bar.set_postfix(loss=loss.item())
        save_checkpoint(model, optimizer, epoch + 1, path=f"unetLMHSA.pth")
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")

        ema.apply()


if __name__ == '__main__':
    main()
