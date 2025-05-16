import numpy as np
import os
import pickle
import nimblephysics as nimble

from einops import einsum

from .subject import Subject

class UIPSubject(Subject):
    def get_real_imu_data(self):
        for key in self.trial_map.keys():
            with open(os.path.join(self.corresponding_imu_path, f"{key}.pkl"), "rb") as file:
                acc = pickle.load(file, encoding="latin1")["acc"]

            self.trial_imu_map[key] = {
                "acc": acc.cpu().numpy(),
            }

    def setup_subject(self):
        self.corresponding_imu_path = os.path.join(os.path.dirname(self.b3d_path), f"sensors/S{self.subject_num}")
        self.create_trial_map()
        self.get_joint_data()
        self.get_real_imu_data()

        self.run_first_pass()
        self.run_second_pass()

    def run_second_pass(self):
        self.skeleton.setGravity(-self.gravity_vec_two.astype(np.float64))

        self.imus = {}
        for key in self.trial_map.keys():
            imu_list = []
            for idx, name in enumerate(self.nimble_body_nodes):
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