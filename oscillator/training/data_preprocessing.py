import numpy as np
import torch
from torch.utils.data import Dataset


class OscillatorDataset(Dataset):

    def __init__(self, npz_path):

        data = np.load(npz_path)

        self.obs = data["obs"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        self.next_obs = data["next_obs"].astype(np.float32)
        self.dones = data["dones"].astype(np.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):

        obs = torch.tensor(self.obs[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        next_obs = torch.tensor(self.next_obs[idx], dtype=torch.float32)
        done = torch.tensor(self.dones[idx], dtype=torch.float32)

        return {
            "obs": obs,
            "action": action,
            "next_obs": next_obs,
            "done": done
        }