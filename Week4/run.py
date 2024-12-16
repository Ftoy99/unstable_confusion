import torch
from matplotlib import pyplot as plt

from UNet import UNet


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
