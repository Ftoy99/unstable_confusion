import torch


class Gauss:
    def __init__(self, beta_start, beta_end, T, device):
        """
        Initialize the Gaus class with a given beta schedule and number of timesteps.

        Args:
            T (int): Number of timesteps in the diffusion process.
            schedule (str): Type of beta schedule ("linear" or "cosine").
            beta_start (float): Starting value for beta (used in linear schedule).
            beta_end (float): Ending value for beta (used in linear schedule).
        """
        self.T = T
        self.beta = torch.linspace(beta_start, beta_end, T)

        # Compute alpha values
        self.alpha = 1 - self.beta
        self.alphas_cumprod = torch.cumprod(self.alpha, dim=0)  # \bar{\alpha}_t
        self.sqrt_alpha_cumprod = self.alphas_cumprod.sqrt()  # \sqrt{\bar{\alpha}_t}
        self.sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod).sqrt().to(device=device)  # \sqrt{1 - \bar{\alpha}_t}

    def q_sample(self, x_0, t):
        noise = torch.randn_like(x_0)

        # Gather precomputed values for the given timesteps
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        # Compute the noisy sample
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise, noise
