import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import smooth_l1_loss

NUM_OF_IMUS = 13
NUM_OF_NOISE_PARAMS = 10

class Encoder(nn.Module):
    def __init__(self, d_model, num_encoders=6, feed_forward_dim=2048, dropout=0.1, heads=8):
        super(Encoder, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=heads, activation=F.gelu, dim_feedforward=feed_forward_dim, dropout=dropout, batch_first=True)
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

        d = F.softplus(noise_params[:, 1, :]).view(seq_len, 1, NUM_OF_IMUS)
        k = (d**2).div(4) + F.softplus(noise_params[:, 0, :]).view(seq_len, 1, NUM_OF_IMUS)

        d_theta = F.softplus(noise_params[:, 3, :]).view(seq_len, 1, NUM_OF_IMUS)
        k_theta = (d_theta**2).div_(4) + F.softplus(noise_params[:, 2, :]).view(seq_len, 1, NUM_OF_IMUS)

        noise_bias = noise_params[:, 8, :].T
        noise_std = F.softplus(noise_params[:, 9, :].T)

        kinematics_list = []
        for imu_num in range(NUM_OF_IMUS):
            k_imu = k[:, :, imu_num]
            d_imu = d[:, :, imu_num]
            omega1 = (k_imu.mul_(4) - (d_imu ** 2)).sqrt_() / 2

            exp_term_linear = ((-d_imu / 2) * t_step).exp_()
            sin_term_linear = (t_step * omega1).add_(phi[:, :, imu_num]).sin_()
            linear_kinematics = c[:, :, imu_num] * exp_term_linear * sin_term_linear

            k_theta_imu = k_theta[:, :, imu_num]
            d_theta_imu = d_theta[:, :, imu_num]
            omega1_theta = (k_theta_imu.mul_(4) - (d_theta_imu ** 2)).sqrt_() / 2

            exp_term_angular = ((-d_theta_imu / 2) * t_step).exp_()
            sin_term_angular = (t_step * omega1_theta).add_(phi_theta[:, :, imu_num]).sin_()
            angular_kinematics = c_theta[:, :, imu_num] * exp_term_angular * sin_term_angular

            spring_damper_kinematics_per_step = linear_kinematics.add_(angular_kinematics).triu_()
            summed_kinematics = torch.sum(spring_damper_kinematics_per_step, dim=0, keepdim=True)
            kinematics_list.append(summed_kinematics)

        return torch.cat(kinematics_list, dim=0).add_(min_orig_accel_norm).add_(noise_bias), noise_std



class Trase(nn.Module):
    def __init__(self, d_model, inp_emb_dim, device, num_encoders=6, dim_feed_forward=1024, dropout=0.1, heads=8):
        super(Trase, self).__init__()

        self.linear1 = nn.Linear(inp_emb_dim, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.activation1 = nn.GELU()

        self.linear2 = nn.Linear(d_model, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.activation2 = nn.GELU()

        self.encoder = Encoder(d_model, num_encoders, dim_feed_forward, dropout=dropout, heads=heads)

        self.linear3 = nn.Linear(d_model, d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.activation3 = nn.GELU()

        self.noise_regressor = Noise_Regressor(d_model, device)

    def forward(self, x, min_orig_accel_norm):
        x = self.linear1(x)
        x = self.layer_norm1(x)
        x = self.activation1(x)

        residual_1 = x

        x = self.linear2(x)
        x = self.layer_norm2(x)
        x = self.activation2(x)

        x = residual_1 + x

        encoded_states = self.encoder(x)

        residual_2 = encoded_states

        x = self.linear3(encoded_states)
        x = self.layer_norm3(x)
        x = self.activation3(x)

        encoded_states = residual_2 + x

        kinematics, std = self.noise_regressor(encoded_states, min_orig_accel_norm)

        return kinematics, std # kinematics is essentially the mean

class Trase_Loss(nn.Module):
    def __init__(self, l1_weight=0.5, nll_weight=0.5):
        super(Trase_Loss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.gaussian_negative_log_likelihood = nn.GaussianNLLLoss()
        self.l1_weight = l1_weight
        self.nll_weight = nll_weight

    def forward(self, kinematics, imu, std):
        l1_term = self.l1_weight * self.smooth_l1(kinematics, imu)
        nll_term = self.nll_weight * self.gaussian_negative_log_likelihood(kinematics, imu, std)

        return l1_term + nll_term