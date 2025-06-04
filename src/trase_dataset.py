import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from einops import rearrange


class TraseDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(self.data_path, "rb") as file:
            self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.data[idx]["inputs"]).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        accelerations_output = torch.from_numpy(self.data[idx]["accelerations_output"]).to(device=self.device, dtype=torch.float32).T

        angular_velocities_output = None
        
        if 'angular_velocities_output' in self.data[idx]:
          angular_velocities_numpy = self.data[idx]["angular_velocities_output"]
          angular_velocities_output = torch.from_numpy(angular_velocities_numpy).to(device=self.device, dtype=torch.float32).T if angular_velocities_numpy is not None else None

        output_mask = torch.from_numpy(self.data[idx]["output_mask"]).to(device=self.device, dtype=torch.float32)
        weights = torch.tensor(self.data[idx]["weights"], device=self.device, dtype=torch.float32)

        return {
          "inputs": inputs,
          "accelerations_output": accelerations_output,
          "angular_velocities_output": angular_velocities_output,
          "output_mask": output_mask,
          "weights": weights,
          "origin": self.data[idx]["dataset"]
        }
