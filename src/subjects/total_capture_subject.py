import numpy as np

import os
import pickle
import nimblephysics as nimble
import torch
import theseus as th
from scipy import signal
import matplotlib.pyplot as plt

from einops import einsum, rearrange, repeat

from utils import Minimizer
from .subject import Subject
from .constants import TOTAL_CAPTURE_IMU_LOCATIONS

import torch.nn.init as init


class TotalCaptureSubject(Subject):
    def setup_subject(self):
        self.corresponding_imu_path = os.path.join(os.path.dirname(self.b3d_path), f"sensors/s{self.subject_num}/imu/")
        self.inches_to_meters = 0.0254
        self.create_trial_map()
        self.get_joint_data()
        self.get_real_imu_data()

        self.run_first_pass()
        self.run_second_pass()


    def get_real_imu_data(self):
        imu_data = TotalCaptureSubject.process_all_files(self.corresponding_imu_path)
        
        for key in self.trial_map.keys():
            measurements = np.stack([imu_data[key][part][:, 4:] for part in TOTAL_CAPTURE_IMU_LOCATIONS])

            acc = rearrange(measurements[..., :3], "imu seq meas -> seq imu meas")
            ang_vel = rearrange(measurements[..., 3:6], "imu seq meas -> seq imu meas")
            magnetometer = rearrange(measurements[..., 6:], "imu seq meas -> seq imu meas")
            
            self.trial_imu_map[key] = {
                "acc": acc,
                "ang_vel": ang_vel,
                "magnetometer": magnetometer
            }

    # Override
    def create_trial_map(self):
        for i in range(self.subject.getNumTrials()):
            key = self.subject.getTrialOriginalName(i).split("_")[0]
            curr_trial = self.subject.readFrames(trial=i,
                                                 startFrame=0,
                                                 numFramesToRead=self.subject.getTrialLength(i),
                                                 includeProcessingPasses=True)

            if key in self.trial_map:
                self.trial_map[key].extend(curr_trial)
            else:
                self.trial_map[key] = curr_trial

    def run_second_pass(self):
        self.skeleton.setGravity(-self.gravity_vec_two.astype(np.float64))
        self.imus = {}

        for key in self.trial_map.keys():
            imu_list = []
            for idx, name in enumerate(self.nimble_body_nodes):
                body_node = self.skeleton.getBodyNode(name)
                transform = nimble.math.Isometry3.Identity()
                transform.set_matrix(self.opt_trans[key]["imu_in_body_node_frame"][idx])

                imu_list.append((body_node, transform))
            self.imus[key] = imu_list

        self.get_syn_imu_to_real_transformations()

    # Override
    def run_first_pass(self):
        self.skeleton.setGravity(-self.gravity_vec_two.astype(np.float64))
        self.imus = {key : [(self.skeleton.getBodyNode(name), nimble.math.Isometry3.Identity()) for name in self.nimble_body_nodes] for key in self.trial_map.keys()}

        self.get_syn_imu_to_real_transformations()
        self.sync_real_syn_imu_data()
        self.find_optimal_syn_transformations()

    # Overide

    def sync_real_syn_imu_data(self):
        """
        Synchronizes real and synthetic IMU data for each trial.

        The synchronization is based on maximizing the cross-correlation between
        the magnitudes of acceleration and angular velocity of a specific sensor
        (index 5, assumed to be the left foot).

        The method updates self.trial_imu_map, self.syn_imu, and
        self.joint_data_map with the sliced, synchronized data.
        """
        for key in self.trial_map.keys():
            # Ensure data exists for the key
            if key not in self.trial_imu_map or key not in self.syn_imu or \
               key not in self.joint_data_map:
                print(f"Warning: Missing data for key {key}. Skipping synchronization.")
                continue
            if 'acc' not in self.trial_imu_map[key] or \
               'ang_vel' not in self.trial_imu_map[key] or \
               'acc' not in self.syn_imu[key] or \
               'angular_vel' not in self.syn_imu[key]:
                print(f"Warning: Missing 'acc' or 'ang_vel'/'angular_vel' for key {key}. Skipping.")
                continue


            # Compute magnitudes (T, sensors)
            # Ensure data has at least 3 dimensions for axis=-1 to be valid for typical IMU (frames, sensors, axes)
            # And at least 6 sensors for index 5 to be valid.
            if self.trial_imu_map[key]['acc'].ndim < 2 or self.trial_imu_map[key]['acc'].shape[1] <= 5 or \
               self.syn_imu[key]['acc'].ndim < 2 or self.syn_imu[key]['acc'].shape[1] <= 5 or \
               self.trial_imu_map[key]['ang_vel'].ndim < 2 or self.trial_imu_map[key]['ang_vel'].shape[1] <= 5 or \
               self.syn_imu[key]['angular_vel'].ndim < 2 or self.syn_imu[key]['angular_vel'].shape[1] <= 5:
                print(f"Warning: Insufficient dimensions or sensors for key {key} for norm calculation. Skipping.")
                continue

            real_imu_acc = np.linalg.norm(self.trial_imu_map[key]['acc'], axis=-1)
            syn_imu_acc  = np.linalg.norm(self.syn_imu[key]['acc'], axis=-1)
            real_imu_ang = np.linalg.norm(self.trial_imu_map[key]['ang_vel'], axis=-1)
            syn_imu_ang  = np.linalg.norm(self.syn_imu[key]['angular_vel'], axis=-1)

            # Masks for correlation (unused for slicing length in current form, but kept from original)
            # Ensure masks are based on the correct sensor data if syn_imu_acc/ang could be empty
            if syn_imu_acc.shape[0] == 0 or syn_imu_ang.shape[0] == 0:
                print(f"Warning: Synthetic IMU data for key {key} has zero length. Skipping correlation.")
                # Set empty slices or handle as an error
                i_real = slice(0,0)
                i_syn = slice(0,0)
            elif real_imu_acc.shape[0] == 0 or real_imu_ang.shape[0] == 0:
                print(f"Warning: Real IMU data for key {key} has zero length. Skipping correlation.")
                i_real = slice(0,0)
                i_syn = slice(0,0)
            else:
                mask_acc = np.ones_like(syn_imu_acc[:, 5], dtype=np.float32) # Mask for sensor 5
                mask_ang = np.ones_like(syn_imu_ang[:, 5], dtype=np.float32) # Mask for sensor 5

                def raw_corr(a, b, m):
                    # Ensure a and b are 1D for correlate
                    if a.ndim > 1 or b.ndim > 1 or m.ndim > 1:
                         raise ValueError("Inputs to raw_corr must be 1D arrays.")
                    return np.correlate(a, b * m, mode='full')

                # Use sensor index 5 which corresponds to the left foot for alignment
                corr_acc   = raw_corr(real_imu_acc[:, 5], syn_imu_acc[:, 5], mask_acc)
                corr_ang   = raw_corr(real_imu_ang[:, 5], syn_imu_ang[:, 5], mask_ang)
                corr_total = corr_acc + corr_ang

                if len(corr_total) == 0: # Should not happen if inputs are non-empty
                    best_idx = 0
                else:
                    best_idx   = np.argmax(corr_total)


                # Number of frames and computed lag
                n_real = real_imu_acc.shape[0]
                n_syn  = syn_imu_acc.shape[0]
                lag    = best_idx - (n_syn - 1)

                real_start = max(0, lag)
                real_end = min(n_real, n_syn + lag)

                syn_start = max(0, -lag) # Equivalent to real_start - lag
                syn_end = min(n_syn, n_real - lag) # Equivalent to real_end - lag

                if real_end < real_start: real_end = real_start
                if syn_end < syn_start: syn_end = syn_start

                i_real = slice(real_start, real_end)
                i_syn  = slice(syn_start, syn_end)

            # Apply slices
            # Need to handle cases where original data might be missing these keys
            syn_data_updates = {}
            if 'acc' in self.syn_imu[key]:
                syn_data_updates['acc'] = self.syn_imu[key]['acc'][i_syn]
            if 'angular_vel' in self.syn_imu[key]:
                syn_data_updates['angular_vel'] = self.syn_imu[key]['angular_vel'][i_syn]
            if 'angular_accel' in self.syn_imu[key]: # This was in original code
                 syn_data_updates['angular_accel'] = self.syn_imu[key]['angular_accel'][i_syn]


            trial_data_updates = {}
            if 'acc' in self.trial_imu_map[key]:
                trial_data_updates['acc'] = self.trial_imu_map[key]['acc'][i_real]
            if 'ang_vel' in self.trial_imu_map[key]:
                trial_data_updates['ang_vel'] = self.trial_imu_map[key]['ang_vel'][i_real]
            if 'magnetometer' in self.trial_imu_map[key]: # This was in original code
                trial_data_updates['magnetometer'] = self.trial_imu_map[key]['magnetometer'][i_real]

            joint_data_updates = {}
            # Joint data is (features, time_frames)
            if 'joint_angles' in self.joint_data_map[key]:
                 joint_data_updates['joint_angles'] = self.joint_data_map[key]['joint_angles'][:, i_syn]
            if 'joint_vel' in self.joint_data_map[key]:
                 joint_data_updates['joint_vel'] = self.joint_data_map[key]['joint_vel'][:, i_syn]
            if 'joint_acc' in self.joint_data_map[key]:
                 joint_data_updates['joint_acc'] = self.joint_data_map[key]['joint_acc'][:, i_syn]

            self.trial_imu_map[key].update(trial_data_updates)
            self.syn_imu[key].update(syn_data_updates)
            self.joint_data_map[key].update(joint_data_updates)



    @staticmethod
    def read_sensors_file(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        num_sensors, num_frames = map(int, lines[0].split())
        data = {}

        for frame in range(1, num_frames + 1):
            start_line = 2 + (frame - 1) * (num_sensors + 1)
            for i in range(num_sensors):
                line = lines[start_line + i].split()
                sensor_name = line[0]
                measurements = list(map(float, line[1:]))

                if sensor_name not in data:
                    data[sensor_name] = []
                data[sensor_name].append(measurements)

        return {k: np.array(v) for k, v in data.items()}

    @staticmethod
    def process_all_files(directory):
        all_data = {}
        for filename in os.listdir(directory):
            if filename.endswith('.sensors'):
                file_path = os.path.join(directory, filename)
                file_data = TotalCaptureSubject.read_sensors_file(file_path)
                trial_name = filename.split("_")[0]

                if trial_name not in all_data:
                    all_data[trial_name] = {}

                for sensor, measurements in file_data.items():
                    all_data[trial_name][sensor] = measurements

        return all_data

    # Override
    def find_optimal_syn_transformations(self):
        for trial_name, syn_imu in self.syn_imu.items():
            # Convert numpy arrays to PyTorch tensors
            num_t_steps = syn_imu["acc"].shape[0]
            losses = []

            real_imu_data = torch.from_numpy(self.trial_imu_map[trial_name]["acc"]).to(device=self.device, dtype=torch.float32)
            real_imu_ang_vel_data = torch.from_numpy(self.trial_imu_map[trial_name]["ang_vel"]).to(device=self.device, dtype=torch.float32)
            syn_imu_data = torch.from_numpy(syn_imu["acc"]).to(device=self.device, dtype=torch.float32)
            syn_imu_angular_vel_data = torch.from_numpy(syn_imu["angular_vel"]).to(device=self.device, dtype=torch.float32)
            syn_imu_angular_accel_data = torch.from_numpy(syn_imu["angular_accel"]).to(device=self.device, dtype=torch.float32)

            real_imu_normed = torch.linalg.norm(real_imu_data, dim=-1)
            syn_imu_data_normed  = torch.linalg.norm(syn_imu_data, dim=-1)
            
            real_imu_angv_normed = torch.linalg.norm(real_imu_ang_vel_data, dim=-1)
            syn_imu_angv_data_normed  = torch.linalg.norm(syn_imu_angular_vel_data, dim=-1)
            syn_imu_angular_accel_data_normed = torch.linalg.norm(syn_imu_angular_accel_data, dim=-1)

            high_signal_filter_acc = (real_imu_normed < 15) & (syn_imu_data_normed < 15)
            high_signal_filter_acc = high_signal_filter_acc.float().detach().unsqueeze(-1)
            
            high_signal_filter_angv = (real_imu_angv_normed < 15) & (syn_imu_angv_data_normed < 15) & (syn_imu_angular_accel_data_normed < 30)
            high_signal_filter_angv = high_signal_filter_angv.float().detach().unsqueeze(-1)

            trial_body_node_world_transforms = torch.tensor(self.get_world_transforms(trial_name), device=self.device, dtype=torch.float32)

            radius_mat = torch.empty(
                len(self.nimble_body_nodes), 3,
                device=self.device, dtype=torch.float32,
                requires_grad=True
            )

            init.uniform_(
                radius_mat,
                a=-1e-3,
                b=1e-3
            )

            unnormed_quaternions = torch.empty(
                len(self.nimble_body_nodes), 4,
                device=self.device, dtype=torch.float32,
            )

            unnormed_quaternions[:, 0] = 1.0
            init.uniform_(
                unnormed_quaternions[:, 1:],
                a=-1e-3,
                b=1e-3
            )

            unnormed_quaternions.requires_grad_(True)

            optimizer = torch.optim.AdamW([{'params': [radius_mat], 'weight_decay': 0},
                                           {'params': [unnormed_quaternions], 'weight_decay': 0}
            ], lr=1e-2)


            for iteration in range(700):
                optimizer.zero_grad()

                current_quaternions = unnormed_quaternions / torch.norm(unnormed_quaternions, dim=-1, keepdim=True)

                rotation_transforms = th.SO3(quaternion=current_quaternions).to_matrix()

                radius_mat_batched = repeat(
                    radius_mat,
                    "imu c -> seq imu c",
                    seq=syn_imu_data.shape[0],
                )

                # Synthetic and Real Angular Acceleration and Velocity will be nearly the same
                real_imu_linear_accel = torch.linalg.cross(syn_imu_angular_accel_data, radius_mat_batched, dim = -1)
                real_imu_velocity = torch.linalg.cross(syn_imu_angular_vel_data, radius_mat_batched, dim = -1)
                real_imu_centripetal_accel = torch.linalg.cross(syn_imu_angular_vel_data, real_imu_velocity, dim = -1)

                real_imu_in_syn_frame = einsum(rotation_transforms, real_imu_data, "imu i j, seq imu j -> seq imu i") + real_imu_linear_accel + real_imu_centripetal_accel
                real_imu_ang_vel_in_syn_frame = einsum(rotation_transforms, real_imu_ang_vel_data, "imu i j, seq imu j -> seq imu i")

                # Calculate acceleration and angular velocity losses separately
                acc_loss = torch.norm((real_imu_in_syn_frame - syn_imu_data) * high_signal_filter_acc, p=1,
                                      dim=-1).mean()
                ang_vel_loss = torch.norm(
                    (syn_imu_angular_vel_data - real_imu_ang_vel_in_syn_frame) * high_signal_filter_angv, p=1,
                    dim=-1).mean()

                loss = acc_loss + ang_vel_loss

                losses.append(loss.item())
                loss.backward()
                optimizer.step()


            unnormed_quaternions = unnormed_quaternions.detach().cpu()
            current_quaternions = unnormed_quaternions / torch.norm(unnormed_quaternions, dim=-1, keepdim=True)

            radius_mat_numpy = radius_mat.detach().cpu()

            imu_pos_and_quat_body_frame = torch.cat([radius_mat_numpy, current_quaternions], dim=-1)
            self.opt_trans[trial_name] = {
                "imu_in_body_node_frame": th.SE3(x_y_z_quaternion=imu_pos_and_quat_body_frame, dtype=torch.float32).to_matrix().detach().cpu().numpy(),
                "imu_in_body_node_frame_quat": imu_pos_and_quat_body_frame,
                "body_node_in_world_frame": trial_body_node_world_transforms.detach().cpu().numpy(),
                "losses": losses
            }


    def get_joint_data(self):
        for key, frames in self.trial_map.items():
            joint_angles_raw = np.vstack([frame.processingPasses[0].pos for frame in frames]).T
            joint_vel_raw = np.vstack([frame.processingPasses[0].vel for frame in frames]).T
            joint_acc_raw = np.vstack([frame.processingPasses[0].acc for frame in frames]).T


            # minimizer = Minimizer(joint_angles_raw.shape[1], self.minimizer_regularization, self.minimizer_smoothing)

            """
            self.joint_data_map[key] = {
                "joint_angles": minimizer.minimize(joint_angles_raw),
                "joint_vel": minimizer.minimize(joint_vel_raw),
                "joint_acc": minimizer.minimize(joint_acc_raw),
            }
            """


            self.joint_data_map[key] = {
                "joint_angles":  joint_angles_raw,
                "joint_vel": self.process_joint_angles(joint_vel_raw, std_cutoff=1.5),
                "joint_acc": self.process_joint_angles(joint_acc_raw, std_cutoff=1.5),
            }


            """
            self.joint_data_map[key] = {
                "joint_angles": signal.filtfilt(self.b, self.a, joint_angles_raw, axis=1),
                "joint_vel": signal.filtfilt(self.b, self.a, joint_vel_raw, axis=1),
                "joint_acc": signal.filtfilt(self.b, self.a, joint_acc_raw, axis=1),
            }
            """