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


def main():
    batch_size = 1
    img_channels = 3
    height = 256
    width = 256

    # Input image tensor
    x = torch.randn(batch_size, img_channels, height, width)
    # Timesteps
    t = torch.randint(0, 1000, (batch_size,))

    unet = UNet()
    optimizer = optim.Adam(unet.parameters(), lr=1e-4)
    load_checkpoint("unet.pth", unet, optimizer)
    output = unet(x, t)
    print(output.shape)

    # Convert the output tensor to a valid image
    output_image = output.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
    output_image = (output_image - output_image.min()) / (
            output_image.max() - output_image.min())  # Normalize to [0, 1]
    output_image = output_image.transpose(1, 2, 0)  # Convert to HxWxC format for visualization

    # Plot the image
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
