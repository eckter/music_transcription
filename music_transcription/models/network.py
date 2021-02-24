import torch
import numpy as np

from ..loader.preprocess import freq_depth
from .utils import GaussianNoise, Clamp


samples_per_out = 32


class Network(torch.nn.Module):
    def __init__(self, device="cuda", scaling_mean=1, scaling_std=1, onset_only=True):
        super(Network, self).__init__()

        last_layer_depth = 1 if onset_only else 49

        self.scaling_mean = torch.tensor(scaling_mean, dtype=torch.float32).to(device)
        self.scaling_std = torch.tensor(scaling_std, dtype=torch.float32).to(device)

        self.dense = torch.nn.Sequential(
            GaussianNoise(0.2),
            self._layer(freq_depth, 512, 1),
            self._layer(512, 512, 3),
            self._layer(512, 128, 7),
            torch.nn.MaxPool1d(2),
            self._layer(128, 128, 7),
            self._layer(128, 128, 7),
            torch.nn.MaxPool1d(2),
            self._layer(128, 128, 7),
            self._layer(128, 64, 7),
            torch.nn.MaxPool1d(2),
            self._layer(64, 64, 7),
            self._layer(64, 64, 7),
            torch.nn.MaxPool1d(2),
            self._layer(64, 64, 7),
            torch.nn.MaxPool1d(2),
            self._layer(64, 64, 7),
            self._layer(64, 64, 7),
            torch.nn.Conv1d(64, last_layer_depth, 1, padding=0, bias=True),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        with torch.no_grad():
            x = (x - self.scaling_mean) / self.scaling_std

        x = self.dense(x.transpose(1, 2)).transpose(1, 2)
        x = self.sigmoid(x)
        x = Clamp().apply(x)
        return x

    @staticmethod
    def _layer(in_depth, k,
               kernel=3,
               stride=1,
               batchnorm=True,
               pad="zeros",
               conv=torch.nn.Conv1d,
               bn=torch.nn.BatchNorm1d):
        if type(kernel) == int:
            padding = ((kernel - 1) // 2)
        else:
            padding = list(((np.array(kernel) - 1) // 2))

        c = conv(in_depth, k,
                 kernel_size=kernel,
                 stride=stride,
                 padding=padding,
                 bias=(not batchnorm),
                 padding_mode=pad)

        if batchnorm:
            return torch.nn.Sequential(
                bn(in_depth),
                c,
                torch.nn.LeakyReLU(0.2)
            )
        else:
            return torch.nn.Sequential(
                c,
                torch.nn.LeakyReLU(0.2)
            )
