import torch
from torch.utils.data import Dataset
import numpy as np
import pickle


class TraseDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(self.data_path, "rb") as file:
            self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.data_path[idx]["inputs"]).to(device=self.deivce, dtype=torch.float32)
        accelerations_output = torch.from_numpy(self.data_path[idx]["accelerations_output"]).to(device=self.deivce, dtype=torch.float32)

        angular_velocities_numpy = self.data_path[idx]["angular_velocites_output"]
        angular_velocities_output = torch.from_numpy(angular_velocities_numpy).to(device=self.deivce, dtype=torch.float32) if angular_velocities_numpy!= None else None

        return inputs, accelerations_output, angular_velocities_output
