import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


NUM_OF_IMUS = 13
NUM_OF_NOISE_PARAMS = 9

class Encoder(nn.Module):
    def __init__(self, d_model, num_encoders=6):
        super(Encoder, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=8, activation=F.gelu)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_encoders)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        residual = x

        x_norm = self.norm1(x)
        x_trans = self.transformer_encoder(x_norm)

        x_norm_2 = self.norm2(x_trans)
        ff_out = self.feed_forward(x_norm_2)

        return ff_out + residual


class Noise_Regressor(nn.Module):
    def __init__(self, d_model, device):
        super(Noise_Regressor, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.hidden_state_to_noise_params = nn.Linear(d_model, NUM_OF_IMUS * NUM_OF_NOISE_PARAMS)
        self.eps = 1e-5
        self.device = device

    """
        hidden_states should be of dimension (Batch, Sequence Len, Dim)
        B should always be 1
    """
    # PLEASE CONVERT TO EINOPS (this will be hardly readable to anyone but the people who live in my head)
    def forward(self, hidden_states, min_orig_accel_norm):
        seq_len = hidden_states.shape[1]

        t_step = torch.triu(torch.arange(seq_len, device=self.device, dtype=torch.float32) - torch.arange(seq_len, device=self.device, dtype=torch.float32)[:, None])

        hidden_normed = self.norm1(hidden_states)
        noise_params = self.hidden_state_to_noise_params(hidden_normed).view(seq_len, NUM_OF_NOISE_PARAMS, NUM_OF_IMUS)

        c = noise_params[:, 4, :].view(seq_len, 1, NUM_OF_IMUS)
        c_theta = noise_params[:, 5, :].view(seq_len, 1, NUM_OF_IMUS)
        phi = noise_params[:, 6, :].view(seq_len, 1, NUM_OF_IMUS)
        phi_theta = noise_params[:, 7, :].view(seq_len, 1, NUM_OF_IMUS)

        d = ((noise_params[:, 1, :] ** 2).add(self.eps)).sqrt_().view(seq_len, 1, NUM_OF_IMUS)
        k = (d**2).div(4) + F.softplus(noise_params[:, 0, :]).view(seq_len, 1, NUM_OF_IMUS)

        d_theta = ((noise_params[:, 3, :] ** 2).add_(self.eps)).sqrt_().view(seq_len, 1, NUM_OF_IMUS)
        k_theta = (d_theta**2).div_(4) + F.softplus(noise_params[:, 2, :]).view(seq_len, 1, NUM_OF_IMUS)

        noise_bias = noise_params[:, 8, :].T

        dynamics_list = []
        for imu_num in range(NUM_OF_IMUS):
            k_imu = k[:, :, imu_num]
            d_imu = d[:, :, imu_num]
            omega1 = (k_imu.mul_(4) - (d_imu ** 2)).sqrt_() / 2

            exp_term_linear = ((-d_imu / 2) * t_step).exp_()
            sin_term_linear = (t_step * omega1).add_(phi[:, :, imu_num]).sin_()
            linear_dynamics = c[:, :, imu_num] * exp_term_linear * sin_term_linear

            k_theta_imu = k_theta[:, :, imu_num]
            d_theta_imu = d_theta[:, :, imu_num]
            omega1_theta = (k_theta_imu.mul_(4) - (d_theta_imu ** 2)).sqrt_() / 2

            exp_term_angular = ((-d_theta_imu / 2) * t_step).exp_()
            sin_term_angular = (t_step * omega1_theta).add_(phi_theta[:, :, imu_num]).sin_()
            angular_dynamics = c_theta[:, :, imu_num] * exp_term_angular * sin_term_angular

            spring_damper_dynamics_per_step = linear_dynamics.add_(angular_dynamics).triu_()
            summed_dynamics = torch.sum(spring_damper_dynamics_per_step, dim=0, keepdim=True)
            dynamics_list.append(summed_dynamics)

        return torch.cat(dynamics_list, dim=0).add_(min_orig_accel_norm).add_(noise_bias)



class TimeModel(nn.Module):
    def __init__(self, d_model, inp_emb_dim, device, num_encoders=6):
        super(TimeModel, self).__init__()

        self.linear1 = nn.Linear(inp_emb_dim, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.activation1 = nn.GELU()
        self.encoder = Encoder(d_model, num_encoders)
        self.noise_regressor = Noise_Regressor(d_model, device)

    def forward(self, x, min_orig_accel_norm):
        x = self.linear1(x)
        x = self.layer_norm1(x)
        x = self.activation1(x)
        encoded_states = self.encoder(x)
        dynamics = self.noise_regressor(encoded_states, min_orig_accel_norm)

        return dynamics