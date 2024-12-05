# Based on Auto-Encoding Variational Bayes paper
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        :param input_dim: Dimensions of input
        :param hidden_dim: Size of hidden layer that is fully connected to latent space
        :param latent_dim: Latent variable z
        """
        super(Encoder, self).__init__()
        # Linear = fully connected
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # input layer
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)  # output mean
        self.fc2_log_var = nn.Linear(hidden_dim, latent_dim)  # output log variance

    def forward(self, x):
        h = torch.relu(self.fc1(x))  # ReLu to introduce non-linearity
        z_mean = self.fc2_mu(h)  # mean
        z_log_var = self.fc2_log_var(h)  # log variation
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_reconstructed = torch.sigmoid(self.fc2(h))  # Sigmoid for binary output (normalized pixels)
        return x_reconstructed


class VAEVariationalBayes(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAEVariationalBayes, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.re_parameterize(z_mean, z_log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_log_var

    def re_parameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)  # Compute standard deviation
        eps = torch.randn_like(std)  # Sample from normal distribution
        z = z_mean + eps * std  # Re-parameterization trick
        return z
