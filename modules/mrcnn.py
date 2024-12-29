import torch.nn as nn
import torch
import torch.nn.init
from modules.conv import Conv1dCustom


class MRCNN(nn.Module):
    def __init__(self, shhs: bool):
        super().__init__()
        if shhs is False:
            self._output_n_channels = 128
            self._output_sample_length = 80
        elif shhs is True:
            self._output_n_channels = 128
            self._output_sample_length = 99

        # all padding, bias given in author's github, NOT in paper!
        self.branch1 = nn.Sequential(
            Conv1dCustom(
                in_channels=1,
                out_channels=64,
                kernel_size=50,
                stride=6,
                bias=False,
                padding=24,
            ),
            nn.BatchNorm1d(num_features=64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(p=0.5),
            Conv1dCustom(
                in_channels=64,
                out_channels=128,
                kernel_size=8,
                stride=1,
                bias=False,
                padding=4,
            ),
            nn.BatchNorm1d(num_features=128),
            nn.GELU(),
            Conv1dCustom(
                in_channels=128,
                out_channels=128,
                kernel_size=8,
                stride=1,
                bias=False,
                padding=4,
            ),
            nn.BatchNorm1d(num_features=128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
        )

        self.branch2 = nn.Sequential(
            Conv1dCustom(
                in_channels=1,
                out_channels=64,
                kernel_size=400,
                stride=50,
                bias=False,
                padding=200,
            ),
            nn.BatchNorm1d(num_features=64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(p=0.5),
            Conv1dCustom(
                in_channels=64,
                out_channels=128,
                kernel_size=7,
                stride=1,
                bias=False,
                padding=3,
            ),
            nn.BatchNorm1d(num_features=128),
            nn.GELU(),
            Conv1dCustom(
                in_channels=128,
                out_channels=128,
                kernel_size=7,
                stride=1,
                bias=False,
                padding=3,
            ),
            nn.BatchNorm1d(num_features=128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.last_dropout = nn.Dropout(0.5)

    def forward(self, x):  # x used as input of first branch
        y = x.detach().clone()  # input of second branch
        x = self.branch1(x)
        y = self.branch2(y)
        I = self.last_dropout(torch.cat((x, y), dim=2))
        return I  # testing: I.size:(128,128,80)

    @property
    def output_n_channels(self):
        return self._output_n_channels

    @property
    def output_sample_length(self):
        return self._output_sample_length
