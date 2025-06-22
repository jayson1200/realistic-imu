import torch
import torch.nn as nn
from jaxtyping import Float

from x_transformers.x_transformers import Encoder, FeedForward

class SyntheticMotionEncoder(nn.Module):
    """
    Takes the synthetic motion data and generates a sequence of hidden
    states for each time step
    """
    def __init__(self,
                 input_dim: float,
                 transformer_dim: float,
                 depth: float,
                 dropout: float,
                 mult: float):

        super().__init__()

        self.native_emb_feedforward = FeedForward(
            dim=input_dim,
            dim_out=transformer_dim,
            mult=mult,
            dropout=dropout,
            glu=True,
            swish=True
        )

        self.attention_layers = Encoder(
            dim=transformer_dim,
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
        )

    def forward(self, x: Float[torch.Tensor, "batch seq dim"]) -> Float[torch.Tensor, "batch seq dim"]:
        return self.attention_layers(self.native_emb_feedforward(x))


