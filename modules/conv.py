import torch.nn as nn
import torch
import torch.nn.init

import math
from torch.nn.common_types import _size_1_t
from typing import Union


class Conv1dCustom(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(
            self.weight, mean=0, std=math.sqrt(0.02)
        )  # var=0.02--> std=sqrt(0.02)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            bias=False,
        )

    def forward(self, input):
        output = self.conv(input)
        return output[:, :, : -self.padding]
