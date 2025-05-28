from abc import abstractmethod

import nimblephysics as nimble
import numpy as np
import torch
from einops import einsum, repeat
import theseus as th
from scipy import signal

import time
from tqdm import tqdm
from typing import List
import socket
from utils.Minimizer import Minimizer

from .constants import NIMBLE_BODY_NODES_ALL, NIMBLE_BODY_NODE_WEIGHTS


class Subject:
    def __init__(self, b3d_path, GEOMETRY_PATH, nimble_body_nodes):
        self.subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
        self.skeleton = self.subject.readSkel(processingPass=1, geometryFolder=GEOMETRY_PATH)

        self.b3d_path = b3d_path
        self.subject_num = int(b3d_path.split("/")[-1].split(".")[0][1:])
        self.gui = nimble.NimbleGUI()
        self.GRAVITY = 9.80665
        self.nimble_body_nodes = nimble_body_nodes
        self.index_map = {node: idx for idx, node in enumerate(nimble_body_nodes)}

        self.N_MEAS = len(self.nimble_body_nodes) * 3
        self.trial_map = {}
        self.joint_data_map = {}
        self.trial_imu_map = {} # real accelerations
        self.syn_imu = {} # Map of the synthetic imu measurements
        self.opt_trans = {} # Map of the optimal transformations

        self.gravity_vec_one = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(-1, 1)
        self.gravity_vec_two =  np.array([0.0, self.GRAVITY, 0.0], dtype=np.float32).reshape(-1, 1)

        self.fs = 60
        self.cutoff =  8
        self.b, self.a = signal.butter(10, self.cutoff / (self.fs / 2), btype='lowpass')

        self.minimizer_regularization = 1000
        self.minimizer_smoothing = 2 / ((1/60) ** 2)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.setup_subject()

    @abstractmethod
    def setup_subject(self):
        pass

    """
    The purpose of the first pass is to optimize out the world transforms of the real IMUs,
    """
    def run_first_pass(self):
        self.skeleton.setGravity(self.gravity_vec_one.astype(np.float64))
        self.imus = {key : [(self.skeleton.getBodyNode(name), nimble.math.Isometry3.Identity()) for name in self.nimble_body_nodes] for key in self.trial_map.keys()}

        self.get_syn_imu_to_real_transformations()
        self.find_optimal_syn_transformations()

    """
    The purpose of the second pass is to use the world transform to add gravity back to the 
    real imu data and to recalculate the synthetic imu at the world transform of the real imu 
    """
    @abstractmethod
    def run_second_pass(self):
        pass


    def create_trial_map(self):
        for i in range(self.subject.getNumTrials()):
            nimble_trial_name = self.subject.getTrialName(i)
            key = self.subject.getTrialOriginalName(i).split("segment")[0]
            curr_trial = self.subject.readFrames(trial=i,
                                                 startFrame=0,
                                                 numFramesToRead=self.subject.getTrialLength(i),
                                                 includeProcessingPasses=True)

            if key in self.trial_map:
                self.trial_map[key].extend(curr_trial)
            else:
                self.trial_map[key] = curr_trial

    def process_joint_angles(self, joint_angles, std_cutoff=2.0):
        processed = joint_angles.copy()

        for joint_idx in range(joint_angles.shape[0]):
            # Calculate mean and std for current joint
            joint_mean = np.mean(joint_angles[joint_idx])
            joint_std = np.std(joint_angles[joint_idx])

            # Create mask for outliers
            outlier_mask = np.abs(joint_angles[joint_idx] - joint_mean) > (std_cutoff * joint_std)

            if np.any(outlier_mask):
                # Get indices where there are outliers
                outlier_indices = np.where(outlier_mask)[0]

                # Create interpolation function using non-outlier values
                valid_indices = np.where(~outlier_mask)[0]
                valid_values = joint_angles[joint_idx, valid_indices]

                # Interpolate outlier values
                processed[joint_idx, outlier_indices] = np.interp(
                    outlier_indices,
                    valid_indices,
                    valid_values
                )

        return processed

    def get_joint_data(self):
        for key, frames in self.trial_map.items():
            joint_angles_raw = np.vstack([frame.processingPasses[0].pos for frame in frames]).T
            joint_vel_raw = np.vstack([frame.processingPasses[0].vel for frame in frames]).T
            joint_acc_raw = np.vstack([frame.processingPasses[0].acc for frame in frames]).T


            minimizer = Minimizer(joint_angles_raw.shape[1], self.minimizer_regularization, self.minimizer_smoothing)


            """
            self.joint_data_map[key] = {
                "joint_angles": minimizer.minimize(joint_angles_raw),
                "joint_vel": minimizer.minimize(joint_vel_raw),
                "joint_acc": minimizer.minimize(joint_acc_raw),
            }
            """

            self.joint_data_map[key] = {
                "joint_angles": minimizer.minimize(joint_angles_raw),
                "joint_vel": self.process_joint_angles(joint_vel_raw),
                "joint_acc": self.process_joint_angles(joint_acc_raw),
            }

            """
            self.joint_data_map[key] = {
                "joint_angles": signal.filtfilt(self.b, self.a, joint_angles_raw, axis=1),
                "joint_vel": signal.filtfilt(self.b, self.a, joint_vel_raw, axis=1),
                "joint_acc": signal.filtfilt(self.b, self.a, joint_acc_raw, axis=1),
            }
            """


    @abstractmethod
    def get_real_imu_data(self):
        pass

    def generate_nimble_visualization(self, trial_name):
        def renderBasis(key, p, R, colors: List[List[float]] = None):
            if colors is None:
                colors = [[1., 0., 0., 1.], [0., 1., 0., 1.], [0., 0., 1., 1.]]

            self.gui.nativeAPI().createLine(key + '_x', [p, R[:, 0] / 10 + p], colors[0], width=[1.])
            self.gui.nativeAPI().createLine(key + '_y', [p, R[:, 1] / 10 + p], colors[1], width=[1.])
            self.gui.nativeAPI().createLine(key + '_z', [p, R[:, 2] / 10 + p], colors[2], width=[1.])

        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))  # Bind to a free port assigned by the OS
                return s.getsockname()[1]  # Return the port number

        self.gui.serve(find_free_port())

        transform_dict = self.opt_trans[trial_name]

        for t in tqdm(range(self.joint_data_map[trial_name]["joint_angles"].shape[1])):
            self.skeleton.setPositions(self.joint_data_map[trial_name]["joint_angles"][:, t])

            for imu_num in range(len(self.nimble_body_nodes)):
                # Shape: (Sequence Length) x (IMUs) x (Rows) x (Cols)
                body_node_in_world = transform_dict["body_node_in_world_frame"]

                # Shape: (IMUs) x (Rows) x (Cols)
                imu_in_body_node = transform_dict["imu_in_body_node_frame"]

                # Grab the heads
                body_node_in_world = body_node_in_world[t, imu_num, :, :]
                imu_in_body_node = imu_in_body_node[imu_num, :, :]

                imu_in_world_frame = body_node_in_world @ imu_in_body_node

                renderBasis(key=f"imu number {imu_num}", p=imu_in_world_frame[:-1, -1], R=imu_in_world_frame[:-1, :-1])

            self.gui.nativeAPI().renderSkeleton(self.skeleton)
            time.sleep(1/60)


    # Get estimated homogenous transformations from synthetix to real imu
    def get_syn_imu_to_real_transformations(self):
        for trial_name, trial in self.joint_data_map.items():
            joint_angles = trial["joint_angles"]
            joint_vel = trial["joint_vel"]
            joint_acc = trial["joint_acc"]

            curr_syn_acc = np.empty((self.N_MEAS,  joint_angles.shape[1]))
            curr_syn_angular_vel = np.empty((self.N_MEAS,  joint_angles.shape[1]))

            for t in range(joint_angles.shape[1]):
                self.skeleton.setPositions(joint_angles[:, t])
                self.skeleton.setVelocities(joint_vel[:, t])
                self.skeleton.setAccelerations(joint_acc[:, t])

                curr_syn_acc[:, t] = self.skeleton.getAccelerometerReadings(self.imus[trial_name])
                curr_syn_angular_vel[:, t] = self.skeleton.getGyroReadings(self.imus[trial_name])


            curr_syn_acc = curr_syn_acc.T.reshape(-1, len(self.nimble_body_nodes), 3)
            curr_syn_angular_vel = curr_syn_angular_vel.T.reshape(-1, len(self.nimble_body_nodes), 3)

            curr_syn_angular_accel = np.zeros_like(curr_syn_angular_vel)

            curr_syn_angular_accel[1:-1] = (curr_syn_angular_vel[2:] - curr_syn_angular_vel[:-2]) / (2/60)
            curr_syn_angular_accel[0] = (curr_syn_angular_vel[1] - curr_syn_angular_vel[0]) / (1/60)
            curr_syn_angular_accel[-1] = (curr_syn_angular_vel[-1] - curr_syn_angular_vel[-2]) / (1/60)

            self.syn_imu[trial_name] = {"acc": curr_syn_acc, "angular_vel": curr_syn_angular_vel, "angular_accel": curr_syn_angular_accel}

    def fill_nans(self, acc):
        # Reshape (T, N, C) -> (T, N*C) for simpler 1D column-wise interpolation
        T, N, C = acc.shape
        acc_2d = acc.reshape(T, N * C)

        for col_idx in range(acc_2d.shape[1]):
            col_data = acc_2d[:, col_idx]
            # Find indices of valid and invalid entries
            valid_mask = ~np.isnan(col_data)
            if not np.any(valid_mask):
                col_data[:] = 0.0
                continue
            if np.all(valid_mask):
                continue

            valid_x = np.where(valid_mask)[0]
            valid_y = col_data[valid_mask]
            invalid_x = np.where(~valid_mask)[0]

            # Interpolate
            col_data[invalid_x] = np.interp(invalid_x, valid_x, valid_y)

        acc_filled = acc_2d.reshape(T, N, C)
        return acc_filled

    def get_world_transforms(self, trial_name):
        num_time_steps = self.joint_data_map[trial_name]["joint_angles"].shape[1]
        world_transform = np.empty((num_time_steps, len(self.nimble_body_nodes), 4, 4))


        for t in range(num_time_steps):
            self.skeleton.setPositions(self.joint_data_map[trial_name]["joint_angles"][:, t])

            for idx, body_node_name in enumerate(self.nimble_body_nodes):
                world_transform[t, idx, :, :] = self.skeleton.getBodyNode(body_node_name).getWorldTransform().matrix()

        return world_transform

    def find_optimal_syn_transformations(self):
        for trial_name, syn_imu in self.syn_imu.items():
            # Convert numpy arrays to PyTorch tensors
            num_t_steps = syn_imu["acc"].shape[0]
            losses = []

            real_imu_data = torch.from_numpy(self.trial_imu_map[trial_name]["acc"]).to(device=self.device, dtype=torch.float32)
            syn_imu_data = torch.from_numpy(syn_imu["acc"]).to(device=self.device, dtype=torch.float32)
            syn_imu_angular_vel_data = torch.from_numpy(syn_imu["angular_vel"]).to(device=self.device, dtype=torch.float32)
            syn_imu_angular_accel_data = torch.from_numpy(syn_imu["angular_accel"]).to(device=self.device, dtype=torch.float32)

            real_imu_normed = torch.linalg.norm(real_imu_data, dim=-1)
            syn_imu_data_normed  = torch.linalg.norm(syn_imu_data, dim=-1)

            high_signal_filter = (real_imu_normed < 10) & (syn_imu_data_normed < 10)
            high_signal_filter = high_signal_filter.float().detach().unsqueeze(-1)


            trial_body_node_world_transforms = torch.tensor(self.get_world_transforms(trial_name), device=self.device, dtype=torch.float32)


            radius_mat = torch.rand(len(self.nimble_body_nodes), 3, dtype=torch.float32, device=self.device).requires_grad_()
            unnormed_quaternions = torch.rand(len(self.nimble_body_nodes), 4, dtype=torch.float32, device=self.device).requires_grad_()

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
                    seq=syn_imu_data.shape[0]
                )

                # Synthetic and Real Angular Acceleration and Velocity will be nearly the same
                real_imu_linear_accel = torch.linalg.cross(syn_imu_angular_accel_data, radius_mat_batched, dim = -1)
                real_imu_velocity = torch.linalg.cross(syn_imu_angular_vel_data, radius_mat_batched, dim = -1)
                real_imu_centripetal_accel = torch.linalg.cross(syn_imu_angular_vel_data, real_imu_velocity, dim = -1)

                real_imu_in_syn_frame = einsum(rotation_transforms, real_imu_data, "imu i j, seq imu j -> seq imu i") + real_imu_linear_accel + real_imu_centripetal_accel

                residual = real_imu_in_syn_frame - syn_imu_data
                loss = torch.norm(residual * high_signal_filter, p=1, dim=-1).mean()

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

    def get_subject_data(self):
        trials = []

        for trial in self.opt_trans.keys():
            inputs = []
            output_mask = []
            accelerations_output = []
            angular_velocities_output = []

            time_steps = self.syn_imu[trial]["acc"].shape[0]

            for node in NIMBLE_BODY_NODES_ALL:
                if node in self.index_map:
                    inputs.append(np.ones((time_steps, 1)))
                    inputs.append(self.syn_imu[trial]["acc"][:, self.index_map[node], :])
                    inputs.append(self.syn_imu[trial]["angular_vel"][:, self.index_map[node], :])
                    inputs.append(self.syn_imu[trial]["angular_accel"][:, self.index_map[node], :])

                    repeated_imu_in_body_node_transforms = repeat(self.opt_trans[trial]["imu_in_body_node_frame_quat"][self.index_map[node]], "c -> seq c", seq=time_steps)
                    inputs.append(repeated_imu_in_body_node_transforms)

                    accelerations_output.append(self.trial_imu_map[trial]["acc"][:, self.index_map[node], :])

                    if "ang_vel" in self.trial_imu_map[trial]:
                        angular_velocities_output.append(self.trial_imu_map[trial]["ang_vel"][:, self.index_map[node], :])

                    output_mask.extend([1] * 3)
                else:
                    inputs.append(np.zeros((time_steps, 17)))
                    accelerations_output.append(np.zeros((time_steps, 3)))

                    if "ang_vel" in self.trial_imu_map[trial]:
                        angular_velocities_output.append(np.zeros((time_steps, 3)))

                    output_mask.extend([0] * 3)


            trials.append({
                "inputs": np.concatenate(inputs, axis=-1),
                "accelerations_output": np.concatenate(accelerations_output, axis=-1),
                "angular_velocities_output": np.concatenate(angular_velocities_output, axis=-1) if len(angular_velocities_output) > 0 else None,
                "weights": np.array(NIMBLE_BODY_NODE_WEIGHTS).reshape(1, -1),
                "output_mask": np.array(output_mask).reshape(1, -1),
                "trial_name": trial,
                "subject_num": self.subject_num,
                "dataset": self.__class__.__name__
            })

        return trials