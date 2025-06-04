import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder as XEncoder
from einops import repeat, rearrange
from torch.utils.checkpoint import checkpoint

NUM_OF_IMU_AXES = 72
NUM_OF_NOISE_PARAMS = 12


class Encoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_encoders: int = 4,
        transformer_ff: int = 2048,
        heads: int = 8,
    ):
        super().__init__()
        self.transformer_encoder = XEncoder(
            dim=d_model,
            depth=num_encoders,
            heads=heads,
            ff_mult=4,
            rotary_pos_emb=True,
            attn_flash=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        x_encoded = checkpoint(self.transformer_encoder, x_norm, use_reentrant=False)
        x_norm2 = self.norm2(x_encoded)
        ff_out = self.feed_forward(x_norm2)
        return ff_out + residual


class PerIMUKinematicsGenerator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(
        self,
        k_imu,
        d_imu,
        phi_imu,
        c_imu,
        k_theta_imu,
        d_theta_imu,
        phi_theta_imu,
        c_theta_imu,
        seq_len,
        time_steps_propogate_kinematics,
    ):
        # build time‐step tensor
        t_step = repeat(
            torch.arange(
                time_steps_propogate_kinematics,
                device=self.device,
                dtype=torch.float32,
            ),
            'tstep -> seq tstep',
            seq=seq_len,
        )

        # compute ω₁
        omega1 = (k_imu * 4 - d_imu**2).sqrt() / 2

        # linear part
        exp_term_linear = ((-d_imu / 2) * t_step).exp()
        sin_term_linear = (t_step * omega1 + phi_imu).sin()
        linear_kinematics = c_imu * exp_term_linear * sin_term_linear

        # angular part
        exp_term_angular = ((-d_theta_imu / 2) * t_step).exp()
        omega2 = (k_theta_imu * 4 - d_theta_imu**2).sqrt() / 2
        sin_term_angular = (t_step * omega2 + phi_theta_imu).sin()
        angular_kinematics = c_theta_imu * exp_term_angular * sin_term_angular

        # sum them
        spring_damper_per_step = linear_kinematics + angular_kinematics

        # compute positions
        positions = (
            torch.arange(time_steps_propogate_kinematics, device=self.device, dtype=torch.int64)
            .unsqueeze(0)
            + torch.arange(seq_len, device=self.device, dtype=torch.int64).unsqueeze(1)
        )

        # flatten and mask
        pos_flat = rearrange(positions, 'seq tstep -> (seq tstep)')
        vals_flat = rearrange(spring_damper_per_step, 'seq tstep -> (seq tstep)')
        mask = pos_flat < seq_len
        pos_flat = pos_flat[mask]
        vals_flat = vals_flat[mask]

        # scatter‐add
        out = torch.zeros(seq_len, dtype=vals_flat.dtype, device=self.device)
        out = torch.scatter_add(out, 0, pos_flat, vals_flat)
        return out.unsqueeze(0)


class Noise_Regressor(nn.Module):
    def __init__(self, d_model, device, max_propogation=300):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.hidden_state_to_noise_params = nn.Linear(
            d_model, NUM_OF_IMU_AXES * NUM_OF_NOISE_PARAMS
        )
        self.device = device
        self.max_propogation = max_propogation
        self.kinematics_generator = PerIMUKinematicsGenerator(device)

    def forward(self, hidden_states):
        seq_len = hidden_states.shape[1]
        time_steps_propogate = min(self.max_propogation, seq_len)

        hidden_normed = self.norm1(hidden_states).squeeze()
        noise_params = rearrange(
            self.hidden_state_to_noise_params(hidden_normed),
            'seq (noise imu) -> seq noise imu',
            noise=NUM_OF_NOISE_PARAMS,
            imu=NUM_OF_IMU_AXES,
        )

        c = rearrange(noise_params[:, 4, :], 'seq imu -> seq 1 imu')
        c_theta = rearrange(noise_params[:, 5, :], 'seq imu -> seq 1 imu')
        phi = rearrange(noise_params[:, 6, :], 'seq imu -> seq 1 imu')
        phi_theta = rearrange(noise_params[:, 7, :], 'seq imu -> seq 1 imu')

        d = rearrange(F.softplus(noise_params[:, 1, :]), 'seq imu -> seq 1 imu')
        k = (d**2) / 4 + rearrange(F.softplus(noise_params[:, 0, :]), 'seq imu -> seq 1 imu')

        d_theta = rearrange(F.softplus(noise_params[:, 3, :]), 'seq imu -> seq 1 imu')
        k_theta = (d_theta**2) / 4 + rearrange(
            F.softplus(noise_params[:, 2, :]), 'seq imu -> seq 1 imu'
        )

        acc_base = rearrange(noise_params[:, 8, :], 'seq imu -> imu seq')
        acc_std = rearrange(F.softplus(noise_params[:, 9, :]), 'seq imu -> imu seq')
        gyro_base = rearrange(noise_params[:, 10, :], 'seq imu -> imu seq')
        gyro_std = rearrange(F.softplus(noise_params[:, 11, :]), 'seq imu -> imu seq')

        kinematics_list = []
        for imu_num in range(NUM_OF_IMU_AXES):
            k_imu = k[:, :, imu_num]
            d_imu = d[:, :, imu_num]
            phi_imu = phi[:, :, imu_num]
            c_imu = c[:, :, imu_num]
            d_theta_imu = d_theta[:, :, imu_num]
            k_theta_imu = k_theta[:, :, imu_num]
            phi_theta_imu = phi_theta[:, :, imu_num]
            c_theta_imu = c_theta[:, :, imu_num]

            kin = checkpoint(
                self.kinematics_generator,
                k_imu,
                d_imu,
                phi_imu,
                c_imu,
                k_theta_imu,
                d_theta_imu,
                phi_theta_imu,
                c_theta_imu,
                seq_len,
                time_steps_propogate,
                use_reentrant=False
            )
            kinematics_list.append(kin)

        kinematics = torch.cat(kinematics_list, dim=0)
        return kinematics, acc_base, acc_std, gyro_base, gyro_std


class Trase(nn.Module):
    def __init__(
        self,
        d_model,
        inp_emb_dim,
        device,
        num_encoders=4,
        dim_feed_forward=1024,
        heads=8,
    ):
        super().__init__()
        self.linear1 = nn.Linear(inp_emb_dim, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.activation1 = nn.GELU()

        self.linear2 = nn.Linear(d_model, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.activation2 = nn.GELU()

        self.encoder = Encoder(
            d_model, num_encoders, dim_feed_forward, heads=heads
        )

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
        x = residual_2 + x

        return self.noise_regressor(x)


class TotalVariationLoss(nn.Module):
    """
    Computes the total variation (TV) of an input tensor along its last dimension.
    """
    def __init__(self, p: int = 1, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.p = p
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x[..., 1:] - x[..., :-1]
        tv = diff.abs().pow(self.p)
        if self.reduction == 'mean':
            return tv.mean()
        elif self.reduction == 'sum':
            return tv.sum()
        return tv


class TraseLoss(nn.Module):
    def __init__(self, total_var_weight=1e-2):
        super().__init__()
        self.gyro_loss = nn.GaussianNLLLoss()
        self.acc_loss = nn.GaussianNLLLoss()
        self.tv_loss = TotalVariationLoss()
        self.total_var_weight = total_var_weight

    def forward(
        self,
        kinematics,
        acc_mean,
        acc_std,
        real_acc,
        gyro_mean,
        gyro_std,
        real_gyro,
        include_gyro=False,
    ):
        loss = self.acc_loss(kinematics + acc_mean, real_acc, acc_std)
        loss += self.total_var_weight * self.tv_loss(acc_mean)
        if include_gyro:
            loss += self.gyro_loss(gyro_mean, real_gyro, gyro_std)
        return loss

