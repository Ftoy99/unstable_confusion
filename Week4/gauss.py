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
        self.sqrt_alpha_cumprod = self.alphas_cumprod.sqrt().to(device=device)  # \sqrt{\bar{\alpha}_t}
        self.sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod).sqrt().to(device=device)  # \sqrt{1 - \bar{\alpha}_t}
        self.sqrt_recip_alpha_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alpha_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

    def q_sample(self, x_0, t):
        noise = torch.randn_like(x_0)

        x_0 = x_0.to(self.sqrt_alpha_cumprod.device)
        noise = noise.to(self.sqrt_alpha_cumprod.device)

        # Gather precomputed values for the given timesteps
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        # Compute the noisy sample
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise, noise

    def _extract(self, tensor, t, shape):
        t = torch.tensor([t], device=tensor.device, dtype=torch.long)
        t = t.view(-1)  # Ensure t is a 1D tensor
        out = tensor.gather(0, t)  # Gather along the 0th dimension
        return out.view(-1, *[1] * (len(shape) - 1))

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict the starting image (x_0) from noisy image x_t and predicted noise.

        Args:
            x_t: Noisy image at timestep t.
            t: Current timestep tensor.
            noise: Predicted noise at timestep t.
        Returns:
            Predicted starting image x_0.
        """
        x_t_shape = x_t.shape
        start = self._extract(self.sqrt_recip_alpha_cumprod, t, x_t_shape) * x_t - self._extract(self.sqrt_recipm1_alpha_cumprod, t, x_t_shape) * noise
        return start

    def p_mean_variance(self, x_start, x_t, t):
        x_t_shape = x_t.shape

        # Posterior variance
        posterior_variance = self._extract(self.beta, t, x_t_shape)

        # Mean of the posterior
        model_mean = (
            self._extract(self.alpha, t, x_t_shape) * x_start
            + self._extract(1 - self.alpha, t, x_t_shape) * x_t
        )
        return model_mean, posterior_variance

    def p_sample(self, x_t, t, noise_pred, clip_denoised=True):
        # Predict x_0 (denoised image) from the noise
        print(x_t.device)
        print(t.device)
        print(noise_pred.device)
        x_start = self.predict_start_from_noise(x_t, t, noise_pred)

        if clip_denoised:
            x_start = torch.clamp(x_start, -1.0, 1.0)

        # Compute the posterior mean and variance

        model_mean, posterior_variance = self.p_mean_variance(x_start, x_t, t)

        # Sample noise
        noise = torch.randn_like(x_t)

        # Mask noise for t == 0 (no noise at the final step)
        t_tensor = torch.tensor([t], device=self.sqrt_alpha_cumprod.device, dtype=torch.long)  # Convert t to a tensor
        nonzero_mask = (t_tensor > 0).float().view(-1, 1, 1, 1)  # Apply mask based on t_tensor

        # Compute x_t-1
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise