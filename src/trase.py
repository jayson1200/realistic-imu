import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import smooth_l1_loss
from x_transformers import Encoder as XEncoder

from einops import repeat

NUM_OF_IMU_AXES = 72
NUM_OF_NOISE_PARAMS = 12


class Encoder(nn.Module):
    def __init__(
            self,
            d_model,
            num_encoders: int = 4,
            transformer_ff: int = 2048,
            dropout: float = 0.1,
            heads: int = 8,
            rotary_xpos_scale_base: int = 1024,
    ):
        super().__init__()

        # After I build this I need to figure out how useful long context is to the model
        # If its not then I can use Alibi
        # If it is then I might need to use a different model or implement Rope scaling
        self.transformer_encoder = XEncoder(
            dim  = d_model,
            depth = num_encoders,
            heads = heads,
            ff_mult=4,
            attn_dropout=dropout,
            ff_dropout=dropout,
            attn_sublayer_dropout=dropout,
            ff_sublayer_dropout=dropout,
            layer_dropout=dropout,
            rotary_pos_emb = True,    # enable classic RoPE
            rotary_xpos_scale_base = rotary_xpos_scale_base
        )

        # follow same post‐attention feed‑forward & norms as before
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        residual = x

        # 1) layer‑norm → x‑transformers encoder (self‑attention + FFN)
        x_norm     = self.norm1(x)
        x_encoded  = self.transformer_encoder(x_norm)

        # 2) post‑norm + our custom FF block + residual
        x_norm2 = self.norm2(x_encoded)
        ff_out  = self.feed_forward(x_norm2)
        return ff_out + residual



class Noise_Regressor(nn.Module):
    def __init__(self, d_model, device, max_propogation=600):
        super(Noise_Regressor, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.hidden_state_to_noise_params = nn.Linear(d_model, NUM_OF_IMU_AXES * NUM_OF_NOISE_PARAMS)
        self.eps = 1e-5
        self.device = device
        self.max_propogation = max_propogation

    """
        hidden_states should be of dimension (Batch, Sequence Len, Dim)
        B should always be 1
    """
    def forward(self, hidden_states):
        seq_len = hidden_states.shape[1]
        time_steps_propogate_kinematics = min(self.max_propogation, seq_len)

        t_step = repeat(torch.arange(time_steps_propogate_kinematics, device=self.device, dtype=torch.float32), 't -> seq t', seq=seq_len)

        hidden_normed = self.norm1(hidden_states)
        noise_params = self.hidden_state_to_noise_params(hidden_normed).view(seq_len, NUM_OF_NOISE_PARAMS, NUM_OF_IMU_AXES)

        c = noise_params[:, 4, :].view(seq_len, 1, NUM_OF_IMU_AXES)
        c_theta = noise_params[:, 5, :].view(seq_len, 1, NUM_OF_IMU_AXES)
        phi = noise_params[:, 6, :].view(seq_len, 1, NUM_OF_IMU_AXES)
        phi_theta = noise_params[:, 7, :].view(seq_len, 1, NUM_OF_IMU_AXES)

        d = F.softplus(noise_params[:, 1, :]).view(seq_len, 1, NUM_OF_IMU_AXES)
        k = (d**2).div(4) + F.softplus(noise_params[:, 0, :]).view(seq_len, 1, NUM_OF_IMU_AXES)

        d_theta = F.softplus(noise_params[:, 3, :]).view(seq_len, 1, NUM_OF_IMU_AXES)
        k_theta = (d_theta**2).div_(4) + F.softplus(noise_params[:, 2, :]).view(seq_len, 1, NUM_OF_IMU_AXES)

        acc_base = noise_params[:, 8, :].T
        acc_std = F.softplus(noise_params[:, 9, :].T)

        gyro_base = noise_params[:, 10, :].T
        gyro_std = F.softplus(noise_params[:, 11, :].T)

        kinematics_list = []
        for imu_num in range(NUM_OF_IMU_AXES):
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

            spring_damper_kinematics_per_step = linear_kinematics.add_(angular_kinematics)
            
            kinematics_positions = (torch.arange(time_steps_propogate_kinematics, device=self.device, dtype=torch.int64).unsqueeze(0) 
                                    + torch.arange(seq_len, device=self.device, dtype=torch.int64).unsqueeze(1)) 

            """
            summed_kinematics = torch.zeros(seq_len + time_steps_propogate_kinematics - 1, 
                                            dtype=torch.float32, 
                                            device=self.device)

            summed_kinematics.scatter_add_(0,
                                           kinematics_positions.reshape(-1),
                                           spring_damper_kinematics_per_step.reshape(-1))
        
            kinematics_list.append(summed_kinematics[:seq_len].unsqueeze(0))
            """

            pos_flat  = kinematics_positions.reshape(-1)
            vals_flat = spring_damper_kinematics_per_step.reshape(-1)
            mask      = pos_flat < seq_len
            pos_flat  = pos_flat[mask]
            vals_flat = vals_flat[mask]

            out = torch.zeros(seq_len, dtype=vals_flat.dtype, device=self.device)
            out.scatter_add_(0, pos_flat, vals_flat)

            kinematics_list.append(out.unsqueeze(0))

        return torch.cat(kinematics_list, dim=0), acc_base, acc_std, gyro_base, gyro_std



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

    def forward(self, x):
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

        kinematics, acc_pred, acc_std, gyro_pred, gyro_std = self.noise_regressor(encoded_states)

        return kinematics, acc_pred, acc_std, gyro_pred, gyro_std

class TotalVariationLoss(nn.Module):
    """
    Computes the total variation (TV) of an input tensor along its last dimension:
        TV(x) = sum_i | x[..., i+1] - x[..., i] |
    
    Args:
        p (int, optional): Exponent for the difference. p=1 gives classic TV; p=2 gives squared-TV.
        reduction (str, optional): 'mean' | 'sum' | 'none'.  Specifies how to reduce over batch & positions.
    """
    def __init__(self, p: int = 1, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.p = p
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute discrete differences along last dim
        diff = x[..., 1:] - x[..., :-1]
        # Apply absolute + exponent
        tv = diff.abs().pow(self.p)
        
        if self.reduction == 'mean':
            return tv.mean()
        elif self.reduction == 'sum':
            return tv.sum()
        else:  # 'none'
            return tv



class TraseLoss(nn.Module):
    def __init__(self, total_var_weight=1e-2):
        super(TraseLoss, self).__init__()
        self.gyro_loss = nn.GaussianNLLLoss()
        self.acc_loss = nn.GaussianNLLLoss()
        self.tv_loss = TotalVariationLoss()
        
        self.total_var_weight = total_var_weight

    def forward(self, kinematics, acc_mean, acc_std, real_acc, gyro_mean, gyro_std, real_gyro, include_gyro=False):
        acc_pred = kinematics + acc_mean
        loss = self.acc_loss(acc_pred, real_acc, acc_std) + self.total_var_weight * self.tv_loss(acc_mean)

        if include_gyro:
          loss += self.gyro_loss(gyro_mean, real_gyro, gyro_std)  


        return loss