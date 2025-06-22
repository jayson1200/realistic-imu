import torch
import einx
from jaxtyping import Float, Int
from typing import Tuple


class CosineNoiseScheduler:
    """
    Implements the cosine noise scheduling strategy for Denoising Diffusion Probabilistic Models (DDPM).

    This scheduler defines a continuous cosine-based schedule for the noise variance across timesteps.
    It provides utilities to compute the forward noising process and sample the reverse (denoising) step
    given model predictions.

    Attributes:
        steps_during_training (int): Total number of diffusion timesteps used during training.
        eps (float): Small constant to avoid numerical issues at t = 0.
        device (torch.device): Device on which all tensors are allocated.
        original_image_decay_t (Float[torch.Tensor, "t"]): √ᾱ_t for each timestep t.
        noise_injection_term_t (Float[torch.Tensor, "t"]): √(1 - ᾱ_t) for each timestep t.
        beta_t (Float[torch.Tensor, "t"]): Instantaneous noise variance β_t per step.
        beta_t_tilde (Float[torch.Tensor, "t"]): Posterior variance term β̃_t per step.
    """

    def __init__(self, num_timesteps: int, device: torch.device) -> None:
        """
        Initialize the cosine noise scheduler.

        Args:
            num_timesteps (int): Number of diffusion steps (T).
            device (torch.device): Device for tensor computations.
        """
        self.steps_during_training = num_timesteps
        self.eps = 1e-5
        self.device = device

        (
            _,
            _,
            self.original_image_decay_t,
            self.noise_injection_term_t,
            self.beta_t,
            self.beta_t_tilde,
        ) = self.get_terms_for_arbitrary_continuous_subsequence(num_timesteps)

    def get_alpha_bar(
        self,
        num_timesteps: int,
        t: Float[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch"]:
        """
        Compute the cumulative product ᾱ_t at arbitrary (possibly non-integer) timestep t
        following the cosine schedule.

        ᾱ(t) = cos^2((t/T + ε/(1+ε)) * π/2) / cos^2(ε/(1+ε) * π/2)

        Args:
            num_timesteps (int): Total number of diffusion steps (T).
            t (Float[torch.Tensor, "batch"]): Tensor of timesteps at which to compute ᾱ.

        Returns:
            Float[torch.Tensor, "batch"]: The normalized cumulative noise decay factor ᾱ_t.
        """
        def f(x: Float[torch.Tensor, "batch"]) -> Float[torch.Tensor, "batch"]:
            progress = x / num_timesteps
            shift = self.eps / (1 + self.eps)
            return torch.cos((progress + shift) * (torch.pi / 2)) ** 2

        zero = torch.tensor(0.0, device=self.device)
        return f(t) / f(zero)

    def get_min_max_var(self) -> Tuple[
        Float[torch.Tensor, "t"],
        Float[torch.Tensor, "t"]
    ]:
        """
        Return the per-step instantaneous variance β_t and posterior variance β̃_t.

        Returns:
            Tuple[β_t, β̃_t]: Two 1D tensors of length T:
                - β_t: noise added at each forward step.
                - β̃_t: posterior variance used in the reverse process.
        """
        return self.beta_t, self.beta_t_tilde

    def get_terms_for_arbitrary_continuous_subsequence(
        self,
        num_timesteps: int
    ) -> Tuple[
        Float[torch.Tensor, "t"],
        Float[torch.Tensor, "t"],
        Float[torch.Tensor, "t"],
        Float[torch.Tensor, "t"],
        Float[torch.Tensor, "t"],
        Float[torch.Tensor, "t"]
    ]:
        """
        Compute all core diffusion terms for timesteps 0..T-1 under the cosine schedule.

        Args:
            num_timesteps (int): Number of diffusion steps (T).

        Returns:
            Tuple containing:
            - α_t: instantaneous cosine-decay factor per step (size T).
            - ᾱ_t: cumulative product of α up to t (size T).
            - √ᾱ_t: original image decay coefficient (size T).
            - √(1 - ᾱ_t): noise injection coefficient (size T).
            - β_t = 1 - α_t (size T).
            - β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t) (size T).
        """
        t_range = torch.arange(
            num_timesteps, dtype=torch.float32, device=self.device
        )

        alpha_bars = self.get_alpha_bar(num_timesteps, t_range)

        # instantaneous α_t = ᾱ_t / ᾱ_{t-1}, with ᾱ_{-1}=1
        alpha = alpha_bars[1:] / alpha_bars[:-1]

        original_image_decay_t = torch.sqrt(alpha_bars)
        pseudo_beta_t = 1 - alpha_bars
        noise_injection_term_t = torch.sqrt(pseudo_beta_t)

        beta_t = 1 - alpha
        beta_t_tilde_scalar = (1 - alpha_bars[:-1]) / (1 - alpha_bars[1:])
        beta_t_tilde = beta_t_tilde_scalar * beta_t

        return (
            alpha,
            alpha_bars,
            original_image_decay_t,
            noise_injection_term_t,
            beta_t,
            beta_t_tilde,
        )

    def add_noise(
        self,
        original: Float[torch.Tensor, "batch seq emb"],
        noise: Float[torch.Tensor, "batch seq emb"],
        t: Int[torch.Tensor, "batch"]
    ) -> Tuple[Float[torch.Tensor, "batch seq emb"],
               Float[torch.Tensor, "batch seq emb"]
    ]:
        """
        Apply the forward diffusion step: mix original data with noise at timestep t.

        x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε

        Args:
            original (Float[torch.Tensor, "batch seq emb"]): Clean input x₀.
            noise (Float[torch.Tensor, "batch seq emb"]): Standard normal noise ε.
            t (Int[torch.Tensor, "batch"]): Integer timesteps at which to apply noise.

        Returns:
            Float[torch.Tensor, "batch seq emb"]: Noised data x_t.
        """
        return (
            self.original_image_decay_t[t] * original
            + self.noise_injection_term_t[t] * noise
        ), noise

    def sample_prev_timestep(
        self,
        xt: Float[torch.Tensor, "batch seq emb"],
        noise_pred: Float[torch.Tensor, "batch seq emb"],
        variance_pred: Float[torch.Tensor, "batch seq emb"],
        t: Int[torch.Tensor, "batch"],
        num_timesteps: int
    ) -> Float[torch.Tensor, "batch seq emb"]:
        """
        Compute a sample x_{t-1} from x_t using model's noise prediction and variance.

        Given the predicted noise ε_θ(x_t, t), the backwards mean is:
            µ_t = 1/√α_t ( x_t - (β_t / √(1-ᾱ_t)) ε_θ )

        And we add stochasticity by sampling from N(µ_t, σ²_t I), where σ²_t may be model-predicted.

        Args:
            xt (Float[torch.Tensor, "batch seq emb"]): Current noised data x_t.
            noise_pred (Float[torch.Tensor, "batch seq emb"]): Model's predicted noise ε_θ.
            variance_pred (Float[torch.Tensor, "batch seq emb"]): Predicted variance σ²_t.
            t (Int[torch.Tensor, "batch"]): Current timesteps.
            num_timesteps (int): Total number of diffusion steps for schedule lookup.

        Returns:
            Float[torch.Tensor, "batch seq emb"]: Sampled data x_{t-1}.
        """

        (
            alpha,
            alpha_bars,
            original_image_decay_t,
            noise_injection_term_t,
            beta_t,
            beta_t_tilde,
        ) = self.get_terms_for_arbitrary_continuous_subsequence(num_timesteps)

        sqrt_alpha = torch.sqrt( alpha)
        mean_factor = 1.0 / sqrt_alpha[t]
        pred_noise_scaled = (beta_t[t] / noise_injection_term_t[t]) * noise_pred

        gaussian_sample = torch.randn_like(xt)
        std_pred = torch.sqrt(variance_pred)
        random_term = std_pred * gaussian_sample

        return mean_factor * (xt - pred_noise_scaled) + random_term