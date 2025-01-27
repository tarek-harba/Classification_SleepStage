import torch
import torch.nn as nn
from modules.conv import Conv1dCustom


class AFR(nn.Module):
    """
    Attention Feature Recalibration (AFR) module. Implemented following authors' description in their paper

    Attributes:
        F (nn.Sequential)
        s (nn.AdaptiveAvgPool1d)
        e (nn.Sequential)
        downsample (nn.Sequential)
        relu (nn.ReLU)
    """

    def __init__(self, input_n_channels, input_sample_length):
        """
        Initializes the AFR module.

        Args:
            input_n_channels (int): Number of input channels.
            input_sample_length (int): Length of the input signal or sequence.
        """
        super().__init__()
        self._output_sample_length = input_sample_length
        # paper: d = number of features
        # input = I.size:(128,128,80)
        # n_channels = L = 128
        self.F = nn.Sequential(
            Conv1dCustom(
                in_channels=input_n_channels,
                out_channels=30,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=30),
            nn.ReLU(),
            Conv1dCustom(
                in_channels=30, out_channels=30, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm1d(num_features=30),
        )  # F.size = 128,30,d=80

        self.s = nn.AdaptiveAvgPool1d(1)

        self.e = nn.Sequential(
            nn.Linear(in_features=30, out_features=1, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=1, out_features=30, bias=False),
            nn.Sigmoid(),
        )

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=30, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features=30),
        )
        self.relu = nn.ReLU()

    def forward(self, I):  # I.size:128,128,80
        """
        Forward pass of the AFR module.

        Args:
            I (torch.Tensor)

        Returns:
            torch.Tensor
        """
        F = self.F(I)
        s = self.s(F)  # s.size:128,30,1
        s_size = list(s.size())
        s = s.view(
            s_size[0], s_size[1]
        )  # s.size:128,30, just 2D to work with FC layers
        e = self.e(s)
        e = e.view(s_size[0], s_size[1], 1)  # back to s.size:128,30,1
        O = torch.mul(F, e)  # O.size:128,30,80
        I = self.downsample(
            I
        )  # downsample I:128,128,80 to I:128,30,80 to get added with O.size:128,30,80
        X = self.relu(I + O)
        return X

    @property
    def output_n_channels(self):
        """
        Number of output channels produced by the AFR module.

        Returns:
            int: Fixed at 30.
        """
        return 30

    @property
    def output_sample_length(self):
        """
        Length of the output sequence.

        Returns:
            int: The length of the output signal.
        """
        return self._output_sample_length
