import torch.nn as nn
import torch
import torch.nn.init
from modules.conv import Conv1dCustom

# input tensor of (128, 3000, 1)
def _validate_input(x):
    """
    Validate input tensor dimensions and type

    Args:
        x (torch.Tensor): Input tensor

    Raises:
        ValueError: If input does not meet specifications
    """
    # Check tensor type
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Input must be a torch.Tensor, got {type(x)}")

    # Check number of dimensions
    if x.ndim != 3:
        raise ValueError(f"Input must be 3D (batch_size, channels, length), got {x.ndim} dimensions")

    # Check number of channels
    if list(x.size())[-1] != 1:
        raise ValueError(f"Input must have 1 channel, got {list(x.size())[-1]} channels")

    # Optional: Check tensor size
    if x.size(0) > 128:
        raise ValueError(f"Input length must be at least 3000, got {x.size(2)}")


class MRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # all padding, bias given in author's github, NOT in paper!
        self.branch1 = nn.Sequential(Conv1dCustom(in_channels=1,out_channels=64,kernel_size=50,stride=6, bias=False,padding=24),
                                     nn.BatchNorm1d(num_features=64),
                                     nn.GELU(),

                                     nn.MaxPool1d(kernel_size=8,stride=2, padding=4),
                                     nn.Dropout(p=0.5),

                                     Conv1dCustom(in_channels=64,out_channels=128,kernel_size=8,stride=1, bias=False,padding=4),
                                     nn.BatchNorm1d(num_features=128),
                                     nn.GELU(),

                                     Conv1dCustom(in_channels=128,out_channels=128,kernel_size=8,stride=1, bias=False, padding=4),
                                     nn.BatchNorm1d(num_features=128),
                                     nn.GELU(),

                                     nn.MaxPool1d(kernel_size=4,stride=4, padding=2))

        self.branch2 = nn.Sequential(Conv1dCustom(in_channels=1, out_channels=64, kernel_size=400, stride=50, bias=False,padding=200),
                                     nn.BatchNorm1d(num_features=64),
                                     nn.GELU(),

                                     nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
                                     nn.Dropout(p=0.5),

                                     Conv1dCustom(in_channels=64, out_channels=128, kernel_size=7, stride=1,bias=False, padding=3),
                                     nn.BatchNorm1d(num_features=128),
                                     nn.GELU(),

                                     Conv1dCustom(in_channels=128, out_channels=128, kernel_size=7, stride=1,bias=False, padding=3),
                                     nn.BatchNorm1d(num_features=128),
                                     nn.GELU(),

                                     nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        self.last_dropout = nn.Dropout(0.5)

    def forward(self, x): # x used as input of first branch
        # _validate_input(x)
        y = x.detach().clone() # input of second branch
        x = self.branch1(x)
        y = self.branch2(y)
        I = self.last_dropout(torch.cat((x, y), dim=2))
        return I # testing: I.size:(128,128,80)
