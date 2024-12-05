import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from models.vae import VAEVariationalBayes
import matplotlib.pyplot as plt


def loss_function(x_reconstructed, x, z_mean, z_log_var):
    # Reconstruction loss (binary cross-entropy)
    reconstruction_loss = nn.BCELoss(reduction='sum')(x_reconstructed, x)

    # KL Divergence
    kl_divergence = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

    # Total loss
    total_loss = reconstruction_loss + kl_divergence
    return total_loss


def train(epochs=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the images
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model, optimizer
    input_dim = 28 * 28  # 28x28 images flattened
    hidden_dim = 400  # Hidden layer size
    latent_dim = 20  # Latent space dimension

    vae = VAEVariationalBayes(input_dim, hidden_dim, latent_dim)
    optimizer = Adam(vae.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, input_dim)  # Flatten input images
            optimizer.zero_grad()

            # Forward pass
            x_reconstructed, z_mean, z_log_var = vae(data)

            # Compute loss
            loss = loss_function(x_reconstructed, data, z_mean, z_log_var)

            # Backward pass and optimization
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_loader.dataset)}')
    return vae


def sample_from_vae(vae, num_samples=1, latent_dim=20):
    # Sample from a standard normal distribution
    z = torch.randn(num_samples, latent_dim)

    # Pass the sampled latent vectors through the decoder
    with torch.no_grad():
        x_reconstructed = vae.decoder(z)

    return x_reconstructed


if __name__ == '__main__':
    vae = train(epochs=10)

    # Example usage
    num_samples = 10  # How many samples you want to generate
    latent_dim = 20  # The dimension of the latent space
    generated_samples = sample_from_vae(vae, 10, 20)

    # Assuming the samples are image tensors, you may want to visualize them

    # Visualize the generated images
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(generated_samples[i].view(28, 28).cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.show()
