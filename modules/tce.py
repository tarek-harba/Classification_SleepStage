import torch
import torch.nn as nn
from modules.conv import CausalConv1d
import math


class TCE(nn.Module):  # Input, X:128,30,80
    def __init__(self, input_n_channels : int, input_sample_length : int):
        super().__init__()
        self._output_n_channels = input_n_channels
        self._output_sample_length = input_sample_length
        ########### Branch 1 ##########
        self.b1_conv1 = CausalConv1d(
            in_channels=input_n_channels,
            out_channels=input_n_channels,
            kernel_size=7,
            stride=1,
            dilation=1,
        )
        self.b1_conv2 = CausalConv1d(
            in_channels=input_n_channels,
            out_channels=input_n_channels,
            kernel_size=7,
            stride=1,
            dilation=1,
        )
        self.b1_conv3 = CausalConv1d(
            in_channels=input_n_channels,
            out_channels=input_n_channels,
            kernel_size=7,
            stride=1,
            dilation=1,
        )
        self.b1_sigmoid = nn.Sigmoid()
        self.b1_bnorm1 = nn.BatchNorm1d(num_features=input_n_channels)
        self.b1_linearchain = nn.Sequential(
            nn.Linear(
                in_features=input_n_channels * input_sample_length,
                out_features=input_n_channels * input_sample_length,
            ),
            nn.Dropout(p=0.1),
            nn.Linear(
                in_features=input_n_channels * input_sample_length,
                out_features=input_n_channels * input_sample_length,
            ),
        )
        self.b1_bnorm2 = nn.BatchNorm1d(num_features=input_n_channels)

        ########### Branch 2 ##########
        self.b2_conv1 = CausalConv1d(
            in_channels=input_n_channels,
            out_channels=input_n_channels,
            kernel_size=7,
            stride=1,
            dilation=1,
        )
        self.b2_conv2 = CausalConv1d(
            in_channels=input_n_channels,
            out_channels=input_n_channels,
            kernel_size=7,
            stride=1,
            dilation=1,
        )
        self.b2_conv3 = CausalConv1d(
            in_channels=input_n_channels,
            out_channels=input_n_channels,
            kernel_size=7,
            stride=1,
            dilation=1,
        )
        self.b2_sigmoid = nn.Sigmoid()
        self.b2_bnorm1 = nn.BatchNorm1d(num_features=input_n_channels)
        self.b2_linearchain = nn.Sequential(
            nn.Linear(
                in_features=input_n_channels * input_sample_length,
                out_features=input_n_channels * input_sample_length,
            ),
            nn.Dropout(p=0.1),
            nn.Linear(
                in_features=input_n_channels * input_sample_length,
                out_features=input_n_channels * input_sample_length,
            ),
        )
        self.b2_bnorm2 = nn.BatchNorm1d(num_features=input_n_channels)

        ########### Converge Branches ##########
        self.converge = nn.Linear(
            in_features=input_n_channels * input_sample_length * 2,
            out_features=input_n_channels * input_sample_length,
        )

    def forward(self, X):  # Input, X:128,30,sample_length
        n_channels = self._output_n_channels
        sample_length = self._output_sample_length
        ########## Branch 1 ##########
        output1 = self._branch1(X, n_channels, sample_length)
        ########## Branch 2 ##########
        output2 = self._branch2(X, n_channels, sample_length)
        ################## Converge Branches ##################
        output = self._merge_branches(output1, output2)
        return output

    def _branch1(self, X : torch.Tensor, n_channels : int, sample_length: int) -> torch.Tensor:
        b1_X1 = self.b1_conv1(X)
        b1_X2 = self.b1_conv2(X)
        b1_X3 = self.b1_conv3(X)
        # Equation 7 in paper
        b1_att = self.b1_sigmoid(
            torch.bmm(
                torch.bmm(b1_X1, torch.transpose(b1_X2, 1, 2))
                / math.sqrt(X.size(dim=-1)),
                b1_X3,
            )
        )
        b1_att = self.b1_bnorm1(b1_att + X)
        b1_att = torch.flatten(b1_att, start_dim=1)
        output1 = self.b1_linearchain(b1_att)
        b1_att = torch.unflatten(b1_att, 1, (n_channels, sample_length))
        output1 = torch.unflatten(output1, 1, (n_channels, sample_length))
        output1 = self.b1_bnorm2(b1_att + output1)
        return output1

    def _branch2(self, X : torch.Tensor, n_channels : int, sample_length: int) -> torch.Tensor:
        b2_X1 = self.b2_conv1(X)
        b2_X2 = self.b2_conv2(X)
        b2_X3 = self.b2_conv3(X)
        X1_heads = torch.chunk(b2_X1, 5, dim=2)
        X2_heads = torch.chunk(b2_X2, 5, dim=2)
        X3_heads = torch.chunk(b2_X3, 5, dim=2)

        head_attn = [None] * 5  # n_heads = 5
        for i in range(5):  # n_heads = 5
            # Equation 8 in paper
            head_attn[i] = self.b2_sigmoid(
                torch.bmm(
                    torch.bmm(X1_heads[i], torch.transpose(X2_heads[i], 1, 2))
                    / math.sqrt(X.size(dim=-1)),
                    X3_heads[i],
                )
            )
        b2_att = torch.cat(head_attn, 2)

        b2_att = self.b2_bnorm1(b2_att + X)
        b2_att = torch.flatten(b2_att, start_dim=1)
        output2 = self.b2_linearchain(b2_att)
        b2_att = torch.unflatten(b2_att, 1, (n_channels, sample_length))
        output2 = torch.unflatten(output2, 1, (n_channels, sample_length))
        output2 = self.b2_bnorm2(b2_att + output2)
        return output2

    def _merge_branches(self, output1 : torch.Tensor, output2 : torch.Tensor) -> torch.Tensor:
        output = torch.cat((output1, output2), 1)
        output = torch.flatten(output, start_dim=1)
        output = self.converge(output)
        output = torch.unflatten(output, 1, (self._output_n_channels, -1))
        return output

    @property
    def output_n_channels(self) -> int:
        return self._output_n_channels

    @property
    def output_sample_length(self) -> int:
        return self._output_sample_length