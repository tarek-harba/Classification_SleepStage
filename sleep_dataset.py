import numpy as np
from torch.utils.data import Dataset

class SleepDataset(Dataset):
    def __init__(self, data_path):
        self._x = np.load(data_path)['x']
        self._y = np.load(data_path)['y']

    def __getitem__(self, idx):
        return self._x[idx], self._y[idx]

    def __len__(self):
        return self._x.shape[0]

