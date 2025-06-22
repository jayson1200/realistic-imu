from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from x_transformers.x_transformers import AttentionLayers, FeedForward
from einops import rearrange

class DIME(nn.Module):
    """
    Implementation of the diffusion inertial measurement estimator (DIME) model
    which uses adaptive layernorm conditoned on the denoising step and cross
    attention to the synthetic motion encoder output sequence to predict the added
    noise and variance.
    """
    def __init__(self,
                 input_dim: float,
                 encoder_dim: float,
                 output_dim: float,
                 diffusion_dim: float,
                 depth: float,
                 dropout: float,
                 mult: float):
        super().__init__()

        self.output_dim = output_dim

        self.embed_feedforward = FeedForward(
            dim=input_dim,
            dim_out=diffusion_dim,
            mult=mult,
            dropout=dropout,
            glu=True,
            swish=True
        )

        self.attention_layers = AttentionLayers(
            dim=diffusion_dim,
            rotary_pos_emb=True,
            attn_flash=True,
            ff_glu=True,
            ff_swish=True,
            causal=False,
            rotary_xpos=True,
            rotary_xpos_scale_base=256,
            depth=depth,
            attn_dropout=dropout,
            ff_dropout=dropout,
            cross_attend=True,
            cross_attention_dim=encoder_dim,
            cross_attention_dropout=dropout,
            cross_attention_scale_attn=True,
            cross_attention_residual=True,
            cross_attention_residual_dropout=dropout,
            use_adaptive_layer_norm=True,
            adaptive_condition_mlp=True,
            use_adaptive_layer_scale=True,
            adaptive_layer_norm_dim=diffusion_dim,
            adaptive_layer_scale_dim=diffusion_dim,
        )

        self.distribution_feedforward = FeedForward(
            dim=diffusion_dim,
            dim_out=output_dim,
            mult=mult,
            dropout=dropout,
            glu=True,
            swish=True
        )



    def forward(self,
                current_denoised_state: Float[torch.Tensor, "batch seq diffusion_dim"],
                synthetic_encoder_states: Float[torch.Tensor, "batch seq encoder_dim"],
                time_step_emb: Float[torch.Tensor, "batch diffusion_dim"]) -> Tuple[Float[torch.Tensor, "batch seq terms"], Float[torch.Tensor, "batch seq terms"]]:
        embeds = self.embed_feedforward(current_denoised_state)
        hidden_states = self.attention_layers(embeds,
                                              context=synthetic_encoder_states,
                                              context_mask=None,
                                              cond=time_step_emb)
        noise_and_variance = self.distribution_feedforward(hidden_states)

        noise, variance = rearrange(noise_and_variance,
                                    'batch seq (distribution_dim terms) -> distribution_dim batch seq terms',
                                    distribution_dim=2)

        return noise, variance




