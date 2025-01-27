import numpy as np
from torch.utils.data import Dataset
import torch


class SleepDataset(Dataset):
    def __init__(self, data_path):
        x = torch.Tensor(np.load(data_path)["x"])
        self._y = np.load(data_path)["y"]
        if len(list(x.size())) == 2:  # used to adjust dimensions of shhs dataset
            x = x.unsqueeze(-1)
        self._x = x.numpy()

    def __getitem__(self, idx):
        return self._x[idx], self._y[idx]

    def __len__(self):
        return self._x.shape[0]
