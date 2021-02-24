from torch.utils.data import DataLoader
import numpy as np
import torch
import random
from tqdm import tqdm
from multiprocessing import Array

from .preprocess import freq_depth
from .save_files import load_with_saves


class SongDataset(torch.utils.data.TensorDataset):
    def __init__(self, files, multiplier=1, multithread=False, overfit=False, sample_files=128):
        super(SongDataset).__init__()
        if not multithread:
            multiplier = 1
        self.data = [load_with_saves(f) for f in tqdm(files)]
        self.multiplier = multiplier
        self.sample_size = sample_files
        self.overfit = overfit
        if multithread:
            self.to_multi()

    def __len__(self):
        return len(self.data) * self.multiplier

    def __getitem__(self, index):
        if index < len(self.data) * self.multiplier:
            index = index % len(self.data)
        if self.overfit:
            index = 0
        return self._extract(*self.data[index], size=self.sample_size, overfit=self.overfit)

    def to_multi(self):
        for i, (s, t, g) in enumerate(self.data):
            self.data[i] = (Array('f', s, lock=False), Array('f', t, lock=False), Array('f', g, lock=False))

    @staticmethod
    def _extract(s, beats, size=freq_depth, overfit=False):
        s = np.array(s).reshape(-1, freq_depth)
        beats = np.array(beats).reshape(s.shape[0], -1)
        mini = 0
        maxi = max(len(s) - size, 0)
        begin = 0 if overfit else random.randint(mini, maxi)
        x = torch.tensor(s[begin:(begin + size)], dtype=torch.float32)
        y = torch.tensor(beats[begin:(begin + size)], dtype=torch.float32)
        return x, y


def get_loader(dataset, workers=16, batch=64):
    timeout = 10 if workers > 0 else 0
    return DataLoader(dataset, num_workers=workers, batch_size=batch, shuffle=True, pin_memory=True, timeout=timeout)
