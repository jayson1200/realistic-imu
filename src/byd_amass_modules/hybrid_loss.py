import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from typing import Tuple

from cosine_scheduler import CosineNoiseScheduler

class HybridLoss(nn.Module):
    """
    PyTorch module for the Improved DDPM hybrid loss, combining
    the simple noise-prediction term with the variational lower-bound.

    L_hybrid = L_simple + λ * L_vlb, where
      • L_simple = E[‖ε – εθ(x_t, t)‖²]
      • L_vlb    = E[KL(q(x_{t-1}|x_t,x₀) ‖ pθ(x_{t-1}|x_t))]
    """

    def __init__(
            self,
            scheduler: CosineNoiseScheduler,
            lambda_vlb: float = 0.001,
            reduction: str = "mean"
    ) -> None:
        """
        Args:
            scheduler: a CosineNoiseScheduler instance
            lambda_vlb: weight λ for the VLB term (typically 1e-3)
            reduction: 'mean' or 'sum' for both MSE losses
        """
        super().__init__()
        self.scheduler = scheduler
        self.lambda_vlb = lambda_vlb
        self.reduction = reduction

        (
            self.alpha,
            self.alpha_bar,
            self.sqrt_alpha_bar,
            self.sqrt_one_minus_alpha_bar,
            self.beta_t,
            self.beta_tilde_t,
        ) = self.scheduler.get_terms_for_arbitrary_continuous_subsequence(
            self.scheduler.steps_during_training
        )

        # Pre-calculate prev_alpha_bar
        self.prev_alpha_bar = torch.cat([
            torch.ones_like(self.alpha_bar[:1]),
            self.alpha_bar[:-1]
        ], dim=0)

        # Pre-calculate mu_tilde coefficients
        self.mu_tilde_x0_coeff = torch.sqrt(self.prev_alpha_bar) * (self.beta_t / (1 - self.alpha_bar))
        self.mu_tilde_xt_coeff = torch.sqrt(self.alpha * (1 - self.prev_alpha_bar)) / (1 - self.alpha_bar)

    def forward(
            self,
            xt: Float[torch.Tensor, "batch ..."],
            noise_pred: Float[torch.Tensor, "batch ..."],
            noise_truth: Float[torch.Tensor, "batch ..."],
            variance_pred: Float[torch.Tensor, "batch ..."],
            x0: Float[torch.Tensor, "batch ..."],
            t: Int[torch.Tensor, "batch"],
    ) -> torch.Tensor:
        L_simple = F.mse_loss(noise_pred, noise_truth, reduction=self.reduction)

        # VLB loss
        sqrt_alpha = torch.sqrt(self.alpha[t])
        mu_theta = (
                           1.0 / sqrt_alpha
                   ) * (
                           xt - (self.beta_t[t] / self.sqrt_one_minus_alpha_bar[t]) * noise_pred
                   )

        # True posterior mean
        mu_tilde = self.mu_tilde_x0_coeff[t] * x0 + self.mu_tilde_xt_coeff[t] * xt

        # KL divergence between N(μ̃, β̃_t) and N(μθ, σ²_pred)
        log_ratio = torch.log(variance_pred) - torch.log(self.beta_tilde_t[t])
        kl = 0.5 * (
                log_ratio
                + (self.beta_tilde_t[t] + (mu_tilde - mu_theta).pow(2)) / variance_pred
                - 1.0
        )
        L_vlb = kl.mean()

        return L_simple + self.lambda_vlb * L_vlb
