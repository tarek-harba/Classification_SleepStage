import torch.nn as nn
import torch
import torch.nn.init

import math
from torch.nn.common_types import _size_1_t
from typing import Union


class Conv1dCustom(nn.Conv1d):
    """
    Custom 1D Convolutional Layer with Gaussian Weight Initialization.

    This class extends the standard `torch.nn.Conv1d` and overrides the weight
    initialization method to use a Gaussian distribution with a mean of 0 and
    a standard deviation derived from the variance (0.02).

    Attributes:
        All attributes are inherited from `torch.nn.Conv1d`.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (_size_1_t): Size of the convolutional kernel.
        stride (_size_1_t, optional): Stride of the convolution. Default is 1.
        padding (Union[str, _size_1_t], optional): Padding added to both sides of the input. Default is 0.
        dilation (_size_1_t, optional): Spacing between kernel elements. Default is 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Default is `True`.
        padding_mode (str, optional): Padding mode. Options are `'zeros'`, `'reflect'`, `'replicate'`, or `'circular'`. Default is `'zeros'`.
        device (torch.device, optional): Specifies the device for model parameters. Default is `None`.
        dtype (torch.dtype, optional): Specifies the data type for model parameters. Default is `None`.
    """

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
        """
        Overrides the default parameter initialization to apply Gaussian
        initialization to the convolutional layer weights.

        Weights are initialized with a mean of 0 and a standard deviation of
        sqrt(0.02), corresponding to a variance of 0.02.
        """
        torch.nn.init.normal_(
            self.weight, mean=0, std=math.sqrt(0.02)
        )  # var=0.02--> std=sqrt(0.02)


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolutional Layer.

    Implements a causal convolution where the output at a time step `t` depends
    only on the input at the same or earlier time steps. This is achieved by adding
    padding to the input tensor such that no future information is used in the convolution.

    Attributes:
        padding (int): Amount of padding added to the input tensor to ensure causality.
        conv (nn.Conv1d): Standard 1D convolutional layer with specified padding.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int, optional): Stride of the convolution. Default is 1.
        dilation (int, optional): Spacing between kernel elements. Default is 1.
    """

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
        """
        Forward pass of the causal convolution.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length)
        """
        output = self.conv(input)
        return output[:, :, : -self.padding]
