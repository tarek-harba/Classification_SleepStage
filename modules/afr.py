import torch
import torch.nn as nn
from modules.conv import Conv1dCustom


class AFR(nn.Module):
    def __init__(self):
        super().__init__()
        # paper: d = number of features
        #input = I.size:(128,128,80)
        # n_channels = L = 128
        self.F = nn.Sequential(Conv1dCustom(in_channels=128,out_channels=30,kernel_size=1,stride=1, bias=False),
                               nn.BatchNorm1d(num_features=30),
                               nn.ReLU(),

                               Conv1dCustom(in_channels=30, out_channels=30, kernel_size=1, stride=1, bias=False),
                               nn.BatchNorm1d(num_features=30)) # F.size = 128,30,d=80

        self.s = nn.AdaptiveAvgPool1d(1)

        self.e = nn.Sequential(nn.Linear(in_features=30, out_features=1, bias=False),
                               nn.ReLU(),
                               nn.Linear(in_features=1, out_features=30, bias=False),
                               nn.Sigmoid())

        self.downsample = nn.Sequential(nn.Conv1d(in_channels=128,
                                                  out_channels=30,
                                                  kernel_size=1,
                                                  bias=False),
                                        nn.BatchNorm1d(num_features=30),
            )
        self.relu = nn.ReLU()

    def forward(self, I): # I.size:128,128,80
        F = self.F(I)
        s = self.s(F) #s.size:128,30,1
        s = s.view(128,30)  #s.size:128,30, just 2D to work with FC layers
        e = self.e(s)
        e = e.view(128, 30, 1)
        O = torch.mul(F, e) #O.size:128,30,80
        I = self.downsample(I) # downsample I:128,128,80 to I:128,30,80 to get added with O.size:128,30,80
        X = self.relu(I + O)
        return X

