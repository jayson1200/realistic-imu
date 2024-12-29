import copy
import os
import sys

import pandas as pd
import torch
from IPython.core.display_functions import clear_output
import quaternion
from torch.utils.data import Dataset

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import numpy as np
import pickle
import nimblephysics as nimble
from scipy.signal import convolve2d


class MotionDataset(Dataset):
    def __init__(self, data_folder, subjects=None, dataset_type="train", minimize=True):
        self.data_folder = data_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_subjects = ["s1", "s2", "s3", "s4"] if not subjects else subjects
        self.gravity = -9.8065
        self.inches_to_meters = 1 / 39.3700787402
        self.dt = 1 / 60
        self.type = dataset_type 
        self.MOCAP_PARTS = [
                    "Head", "Spine3", "Hips", "LeftArm", "RightArm", 
                    "LeftForeArm", "RightForeArm", "LeftUpLeg", "RightUpLeg",
                    "LeftLeg", "RightLeg", "LeftFoot", "RightFoot",
                    "Spine", "Spine1", "Spine2", "LeftHand", "RightHand", "Neck", "LeftShoulder", "RightShoulder" # Values that aren't used 
        ]
        self.minimize = minimize
        self.trial_order = []
        
        self.IMU_PARTS = [
            "Head",
            "Sternum",
            "Pelvis",
            "L_UpArm",
            "R_UpArm",
            "L_LowArm",
            "R_LowArm",
            "L_UpLeg",
            "R_UpLeg",
            "L_LowLeg",
            "R_LowLeg",
            "L_Foot",
            "R_Foot"
        ] 
        
        # For testing purposes
        # self.MOCAP_PARTS = ["LeftFoot"]
        # self.IMU_PARTS = ["L_Foot"]
        dataset_bin_path = os.path.join(self.data_folder, f"dataset_{self.type}.pkl")
        
        if not os.path.exists(dataset_bin_path):
            self.central_acc_kernel = np.array([-1/12, 4/3, -5/2, 4/3, -1/12]).reshape(5, 1) / self.dt ** 2
            self.central_angular_accel_kernel = np.array([1, 0, -1]).reshape(3, 1) / (2 * self.dt)
            
            self.mocap_data = []
            self.imu_data = []
            
            for subject in tqdm(self.target_subjects, desc="Subjects", position=0):
                mocap_path = os.path.join(self.data_folder, subject, "mocap")
                avail_trials = [d for d in os.listdir(mocap_path) if os.path.isdir(os.path.join(mocap_path, d))]
                
                # For testing purposes
                # avail_trials = ["freestyle3"]
                
                for trial in tqdm(avail_trials, desc=f"Trials Completed on {subject}", position=1):
                    mocap_dict, imu_dict = self.get_mocap_imu(trial, subject)
                    self.trial_order.append((subject, trial))
                    
                    # Funny things always seem to happen at the end and the beginning so I removed the first and last 3 indices
                    curr_mocap = np.concatenate([mocap_dict[part] for part in self.MOCAP_PARTS], axis = 1)[3:-3, :]
                    curr_imu = np.concatenate([imu_dict[part] for part in self.IMU_PARTS], axis = 1)[5:-5, :]
                    
                    # Who cares about actually making sure its float32 from the start
                    self.mocap_data.append(curr_mocap.astype(np.float32))
                    self.imu_data.append(curr_imu.astype(np.float32))

            with open(dataset_bin_path ,'wb') as f:
                pickle.dump({"mocap": self.mocap_data, "imu": self.imu_data, "order": self.trial_order}, f)
            
            if 'ipykernel' in sys.modules:
                clear_output(wait=True)
            else:
                os.system('cls' if os.name == 'nt' else 'clear')
        else:
            with open(dataset_bin_path, 'rb') as f:
                data = pickle.load(f)
                
                self.mocap_data = data['mocap']
                self.imu_data = data['imu']
                self.trial_order = data['order']
    
    
    def get_corresponding_mocap_imu_idx(self, idx):
        return self.corresponding_mocap_imu
    
    def __len__(self):
        return len(self.mocap_data)
    
    def get_mocap_imu(self, trial_name: str, subject_name: str):
        imu_data_file = os.path.join(self.data_folder, subject_name, "imu", f"{subject_name}_{trial_name}_Xsens_AuxFields.sensors")
        mocap_glb_ori = os.path.join(self.data_folder, subject_name, "mocap", trial_name, "gt_skel_gbl_ori.txt")
        mocap_glb_pos = os.path.join(self.data_folder, subject_name, "mocap", trial_name, "gt_skel_gbl_pos.txt")
        
        # Reads the sensor files
        with open(imu_data_file, 'r') as f:
            lines = f.readlines()

        num_sensors, num_frames = map(int, lines[0].split())
        imu_data = {}

        for frame in range(1, num_frames + 1):
            start_line = 2 + (frame - 1) * (num_sensors + 1)
            for i in range(num_sensors):
                line = lines[start_line + i].split()
                sensor_name = line[0]
                measurements = list(map(float, line[5:]))

                if sensor_name not in imu_data:
                    imu_data[sensor_name] = np.empty((num_frames , 1))
                
                # Linear Acceleration
                imu_data[sensor_name][frame - 1] = np.linalg.norm(measurements[:3])
            
        mo_ori_glb_df = pd.read_csv(mocap_glb_ori, sep='\t').iloc[:, :-1]
        mo_pos_glb_df = pd.read_csv(mocap_glb_pos, sep='\t').iloc[:, :-1]

        map_func = lambda x: list(map(np.float32, x.split()))
        mo_pos_glb_df = mo_pos_glb_df.map(map_func)
        mo_ori_glb_df = mo_ori_glb_df.map(map_func)
        
        mocap_data = {}
        
        for mocap_tgt_part in self.MOCAP_PARTS:
            right_arm_ori_numpy = np.array(mo_ori_glb_df.loc[:, mocap_tgt_part].to_list())
            right_arm_ori = quaternion.as_quat_array(right_arm_ori_numpy)

            right_arm_pos = np.array(mo_pos_glb_df.loc[:, mocap_tgt_part].to_list())
            right_arm_pos = np.concatenate((np.zeros((right_arm_pos.shape[0], 1)), right_arm_pos), axis=1)

            # Finite differencing
            glb_acc = convolve2d(right_arm_pos, self.central_acc_kernel, mode="valid") * self.inches_to_meters
            glb_acc[:, 2] = self.gravity - glb_acc[:, 2]
            loc_acc = right_arm_ori[2:-2] * quaternion.as_quat_array(glb_acc) * right_arm_ori[2:-2].conjugate()

            past = right_arm_ori_numpy[:-2]
            present = right_arm_ori[1:-1]
            future = right_arm_ori_numpy[2:]

            x_colm_ord = [1, 0, 3, 2]
            y_colm_ord = [2, 3, 0, 1]
            z_colm_ord = [3, 2, 1, 0]

            angx_perm = copy.deepcopy(future)
            angx_perm = angx_perm[:, x_colm_ord]
            angx_perm[:, [1, 2]] *= -1

            angy_perm = copy.deepcopy(future)
            angy_perm = angy_perm[:, y_colm_ord]
            angy_perm[:, [2, 3]] *= -1

            angz_perm = copy.deepcopy(future)
            angz_perm = angz_perm[:, z_colm_ord]
            angz_perm[:, [1, 3]] *= -1

            # In the future when you forget how this works remember that this is not a mistake (https://mariogc.com/post/angular-velocity-quaternions/)
            angv_x = np.einsum("ij, ij -> i", past, angx_perm) / self.dt
            angv_y = np.einsum("ij, ij -> i", past, angy_perm) / self.dt
            angv_z = np.einsum("ij, ij -> i", past, angz_perm) / self.dt

            glb_angv = np.empty(past.shape, dtype=np.float32)
            glb_angv[:, 0] = 0
            glb_angv[:, 1] = angv_x
            glb_angv[:, 2] = angv_y
            glb_angv[:, 3] = angv_z

            glb_angv = quaternion.as_quat_array(glb_angv)
            loc_angv = present * glb_angv * present.conjugate()
            
            loc_acc = quaternion.as_float_array(loc_acc)[:, 1:]
            loc_angv = quaternion.as_float_array(loc_angv)[:, 1:]
            
            loc_ang_accel = convolve2d(loc_angv, self.central_angular_accel_kernel, mode="valid")

            norm_mocap_acc = np.linalg.norm(loc_acc, axis=1)
            norm_mocap_ang_accel = np.linalg.norm(loc_ang_accel, axis=1)

            mocap_data[mocap_tgt_part] = np.empty((norm_mocap_acc.shape[0], 2))

            if self.minimize:
                acc_minimizer = nimble.utils.AccelerationMinimizer(numTimesteps=norm_mocap_acc.shape[0], smoothingWeight = 60 ** 2, regularizationWeight=1000, numIterations=10000)
                ang_acc_minimizer = nimble.utils.AccelerationMinimizer(numTimesteps=norm_mocap_ang_accel.shape[0], smoothingWeight = 60 ** 2, regularizationWeight=1000, numIterations=10000)

                mocap_data[mocap_tgt_part][:, 0] = acc_minimizer.minimize(norm_mocap_acc)
                mocap_data[mocap_tgt_part][:, 1] = ang_acc_minimizer.minimize(norm_mocap_ang_accel)
            else:
                mocap_data[mocap_tgt_part][:, 0] = norm_mocap_acc
                mocap_data[mocap_tgt_part][:, 1] = norm_mocap_ang_accel

        return mocap_data, imu_data
        

    def __getitem__(self, idx):
        return self.mocap_data[idx], self.imu_data[idx]