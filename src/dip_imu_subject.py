import nimblephysics as nimble
import numpy as np
import torch
from einops import einsum, repeat
import theseus as th
from scipy import signal

import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
from typing import List

from scipy.spatial.transform import Rotation


class DIPSubject:
    def __init__(self, b3d_path, GEOMETRY_PATH):
        self.subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
        self.skeleton = self.subject.readSkel(processingPass=1, geometryFolder=GEOMETRY_PATH)
        self.subject_num = int(b3d_path.split("/")[-1].split(".")[0][1:])
        self.gui = nimble.NimbleGUI()
        self.GRAVITY = 9.80665
        self.synthetic_component_names = [
            "head",
            "sternum",
            "pelvis",
            "lshoulder",
            "rshoulder",
            "lupperarm",
            "rupperarm",
            "llowerarm",
            "rlowerarm",
            "lupperleg",
            "rupperleg",
            "llowerleg",
            "rlowerleg",
            "lhand",
            "rhand",
            "lfoot",
            "rfoot",
        ]

        # Corresponding bones to the sensors
        self.skel_names = [
            "head",
            "thorax",
            "pelvis",
            "scapula_l",
            "scapula_r",
            "humerus_l",
            "humerus_r",
            "ulna_l",
            "ulna_r",
            "femur_l",
            "femur_r",
            "tibia_l",
            "tibia_r",
            "hand_l",
            "hand_r",
            "calcn_l",
            "calcn_r"
        ]

        self.N_MEAS = len(self.skel_names) * 3
        self.corresponding_imu_path = os.path.join(os.path.dirname(b3d_path), f"DIP_orig/S{self.subject_num}")
        self.trial_map = {}
        self.joint_data_map = {}
        self.trial_imu_map = {} # real accelerations
        self.syn_imu = {} # Map of the synthetic imu measurements
        self.opt_trans = {} # Map of the optimal transformations

        self.gravity_vec_one = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(-1, 1)
        self.gravity_vec_two =  np.array([0.0, self.GRAVITY, 0.0], dtype=np.float32).reshape(-1, 1)

        self.fs = 60
        self.cutoff =  8
        self.b, self.a = signal.butter(10, self.cutoff / (self.fs / 2), btype='low')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Processing DIP Subject {self.subject_num}")
        self.create_trial_map()
        self.get_joint_data()
        self.get_real_imu_data()

        self.run_first_pass()
        self.run_second_pass()



    """
    The purpose of the first pass is to optimize out the world transforms of the real IMUs,
    """
    def run_first_pass(self):
        self.skeleton.setGravity(self.gravity_vec_one.astype(np.float64))
        self.imus = {key : [(self.skeleton.getBodyNode(name), nimble.math.Isometry3.Identity()) for name in self.skel_names] for key in self.trial_map.keys()}

        self.get_syn_imu_to_real_transformations()
        self.find_optimal_syn_transformations()

    """
    The purpose of the second pass is to use the world transform to add gravity back to the 
    real imu data and to recalculate the synthetic imu at the world transform of the real imu 
    """
    def run_second_pass(self):
        self.skeleton.setGravity(-self.gravity_vec_two.astype(np.float64))

        self.imus = {}
        for key in self.trial_map.keys():
            imu_list = []
            for idx, name in enumerate(self.skel_names):
                body_node = self.skeleton.getBodyNode(name)
                transform = nimble.math.Isometry3.Identity()
                transform.set_matrix(self.opt_trans[key]["body_node_to_imu"][idx])

                imu_list.append((body_node, transform))
            self.imus[key] = imu_list

        self.get_syn_imu_to_real_transformations()

        for trial in self.trial_map.keys():
            imu_in_body_node  =  self.opt_trans[trial]["body_node_to_imu"][:, :-1, :-1]
            body_node_in_world = self.opt_trans[trial]["world_to_body_node"][:, :, :-1, :-1]

            world_to_imu_transform = einsum(imu_in_body_node, body_node_in_world, "imu j i, seq imu k j -> seq imu i k")
            gravity_vec_in_imu_frame = einsum(world_to_imu_transform, self.gravity_vec_two, "seq imu i k, k j-> seq imu i")

            self.trial_imu_map[trial]["acc"] = gravity_vec_in_imu_frame - self.trial_imu_map[trial]["acc"]



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

    def get_joint_data(self):
        for key, frames in self.trial_map.items():
            joint_angles_raw = np.vstack([frame.processingPasses[0].pos for frame in frames]).T
            joint_vel_raw = np.vstack([frame.processingPasses[0].vel for frame in frames]).T
            joint_acc_raw = np.vstack([frame.processingPasses[0].acc for frame in frames]).T

            self.joint_data_map[key] = {
                "joint_angles": signal.filtfilt(self.b, self.a, joint_angles_raw, axis=1),
                "joint_vel": signal.filtfilt(self.b, self.a, joint_vel_raw, axis=1),
                "joint_acc": signal.filtfilt(self.b, self.a, joint_acc_raw, axis=1),
            }


    def get_real_imu_data(self):
        for key, frames in self.trial_map.items():
            with open(os.path.join(self.corresponding_imu_path, f"{key}.pkl"), "rb") as file:
                real_data = pickle.load(file, encoding="latin1")["imu"]
                acc = real_data[:, :, 9:12]
                acc = self.fill_nans(acc)

            self.num_trials = len(self.joint_data_map.keys())

            self.trial_imu_map[key] = {
                "acc": acc,
            }

    def generate_nimble_visualization(self, trial_name):
        self.gui.serve(8080)

        def renderBasis(key, p, R, colors: List[List[float]] = None):
            if colors is None:
                colors = [[1., 0., 0., 1.], [0., 1., 0., 1.], [0., 0., 1., 1.]]

            self.gui.nativeAPI().createLine(key + '_x', [p, R[:, 0] / 10 + p], colors[0], width=[1.])
            self.gui.nativeAPI().createLine(key + '_y', [p, R[:, 1] / 10 + p], colors[1], width=[1.])
            self.gui.nativeAPI().createLine(key + '_z', [p, R[:, 2] / 10 + p], colors[2], width=[1.])

        transform_dict = self.opt_trans[trial_name]

        for t in tqdm(range(self.joint_data_map[trial_name]["joint_angles"].shape[1])):
            self.skeleton.setPositions(self.joint_data_map[trial_name]["joint_angles"][:, t])

            for imu_num in range(len(self.skel_names)):
                # Shape: (Sequence Length) x (IMUs) x (Rows) x (Cols)
                body_node_in_world = transform_dict["world_to_body_node"]

                # Shape: (IMUs) x (Rows) x (Cols)
                imu_in_body_node = transform_dict["body_node_to_imu"]

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


            curr_syn_acc = curr_syn_acc.T.reshape(-1, 17, 3)
            curr_syn_angular_vel = curr_syn_angular_vel.T.reshape(-1, 17, 3)

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
        world_transform = np.empty((num_time_steps, 17, 4, 4))


        for t in range(num_time_steps):
            self.skeleton.setPositions(self.joint_data_map[trial_name]["joint_angles"][:, t])

            for idx, body_node_name in enumerate(self.skel_names):
                world_transform[t, idx, :, :] = self.skeleton.getBodyNode(body_node_name).getWorldTransform().matrix()

        return world_transform

    def find_optimal_syn_transformations(self):
        for trial_name, syn_imu in self.syn_imu.items():
            # Convert numpy arrays to PyTorch tensors
            num_t_steps = syn_imu["acc"].shape[0]
            losses = []

            real_imu_data = torch.tensor(self.trial_imu_map[trial_name]["acc"], device=self.device, dtype=torch.float32)
            syn_imu_data = torch.tensor(syn_imu["acc"], device=self.device, dtype=torch.float32)
            syn_imu_angular_vel_data = torch.tensor(syn_imu["angular_vel"], device=self.device, dtype=torch.float32)
            syn_imu_angular_accel_data = torch.tensor(syn_imu["angular_accel"], device=self.device, dtype=torch.float32)

            real_imu_normed = torch.linalg.norm(real_imu_data, dim=-1)
            syn_imu_data_normed  = torch.linalg.norm(syn_imu_data, dim=-1)

            high_signal_filter = (real_imu_normed < 10) & (syn_imu_data_normed < 10)
            high_signal_filter = high_signal_filter.float().detach().unsqueeze(-1)


            trial_body_node_world_transforms = torch.tensor(self.get_world_transforms(trial_name), device=self.device, dtype=torch.float32)


            radius_mat = torch.rand(17, 3, dtype=torch.float32, device=self.device).requires_grad_()
            unnormed_quaternions = torch.rand(17, 4, dtype=torch.float32, device=self.device).requires_grad_()

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

                real_imu_in_syn_frame = einsum(rotation_transforms, real_imu_data, "imu i j, seq imu i -> seq imu j") + real_imu_linear_accel + real_imu_centripetal_accel

                residual = real_imu_in_syn_frame - syn_imu_data
                loss = torch.norm(residual * high_signal_filter, p=1, dim=-1).mean()

                losses.append(loss.item())
                loss.backward()
                optimizer.step()


            unnormed_quaternions = unnormed_quaternions.detach().cpu()
            current_quaternions = unnormed_quaternions / torch.norm(unnormed_quaternions, dim=-1, keepdim=True)

            radius_mat_numpy = radius_mat.detach().cpu()

            self.opt_trans[trial_name] = {
                "body_node_to_imu": th.SE3(x_y_z_quaternion=torch.cat([-radius_mat_numpy, current_quaternions], dim=-1), dtype=torch.float32).to_matrix().detach().cpu().numpy(),
                "world_to_body_node": trial_body_node_world_transforms.detach().cpu().numpy(),
                "losses": losses
            }