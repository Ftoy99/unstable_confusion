import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.autoencoders import Autoencoder


def main():
    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize model, optimizer, and loss function
    latent_dim = 64
    model = Autoencoder(latent_dim=latent_dim)
    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()  # Set to training mode
        train_loss = 0
        for images, _ in train_loader:
            images = images.view(images.size(0), -1)  # Flatten images
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}")

    # Evaluate on test data
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_images = test_images.view(test_images.size(0), -1)
        reconstructed = model(test_images)

    # Visualize original and reconstructed images
    test_images = test_images.view(-1, 28, 28)
    reconstructed = reconstructed.view(-1, 28, 28)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_images[i].cpu().numpy(), cmap="gray")
        plt.axis("off")

        # Reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].cpu().numpy(), cmap="gray")
        plt.axis("off")
    plt.show()


if __name__ == '__main__':
    main()
