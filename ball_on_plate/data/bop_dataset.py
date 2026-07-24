import torch
from torch.utils.data import Dataset
import numpy as np


class BoPDataset(Dataset):

    def __init__(self, path):

        data = np.load(path)

        self.state = torch.tensor(
            data["state"],
            dtype=torch.float32
        )

        self.action = torch.tensor(
            data["action"],
            dtype=torch.float32
        )

        self.next_state = torch.tensor(
            data["next_state"],
            dtype=torch.float32
        )


    def __len__(self):
        return len(self.state)


    def __getitem__(self, idx):

        return {
            "state": self.state[idx],
            "action": self.action[idx],
            "next_state": self.next_state[idx]
        }