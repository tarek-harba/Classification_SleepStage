import torch
import torch.nn as nn
from modules.conv import Conv1dCustom, CausalConv1d
import math
class TCE(nn.Module):# Input, X:128,30,80
    # if shhs==False: input_length = 80
    # if shhs==True: input_length = 99
    def __init__(self, n_input_channels,sample_length):
        super().__init__()
        ########### Branch 1 ##########
        self.b1_conv1 = CausalConv1d(in_channels=n_input_channels, out_channels=n_input_channels, kernel_size=7, stride=1, dilation=1)
        self.b1_conv2 = CausalConv1d(in_channels=n_input_channels, out_channels=n_input_channels, kernel_size=7, stride=1, dilation=1)
        self.b1_conv3 = CausalConv1d(in_channels=n_input_channels, out_channels=n_input_channels, kernel_size=7, stride=1, dilation=1)
        self.b1_sigmoid = nn.Sigmoid()
        self.b1_bnorm1 =  nn.BatchNorm1d(num_features=n_input_channels)
        self.b1_linearchain = nn.Sequential(nn.Linear(in_features=n_input_channels*sample_length,out_features=n_input_channels*sample_length),
                                            nn.Dropout(p=0.1),
                                            nn.Linear(in_features=n_input_channels*sample_length,out_features=n_input_channels*sample_length))
        self.b1_bnorm2 =  nn.BatchNorm1d(num_features=n_input_channels)

        ########### Branch 2 ##########
        self.b2_conv1 = CausalConv1d(in_channels=n_input_channels, out_channels=n_input_channels, kernel_size=7, stride=1, dilation=1)
        self.b2_conv2 = CausalConv1d(in_channels=n_input_channels, out_channels=n_input_channels, kernel_size=7, stride=1, dilation=1)
        self.b2_conv3 = CausalConv1d(in_channels=n_input_channels, out_channels=n_input_channels, kernel_size=7, stride=1, dilation=1)
        self.b2_sigmoid = nn.Sigmoid()
        self.b2_bnorm1 =  nn.BatchNorm1d(num_features=n_input_channels)
        self.b2_linearchain = nn.Sequential(nn.Linear(in_features=n_input_channels*sample_length,out_features=n_input_channels*sample_length),
                                            nn.Dropout(p=0.1),
                                            nn.Linear(in_features=n_input_channels*sample_length,out_features=n_input_channels*sample_length))
        self.b2_bnorm2 =  nn.BatchNorm1d(num_features=n_input_channels)

        ########### Converge Branches ##########
        self.converge = nn.Linear(in_features=n_input_channels*sample_length*2,
                                  out_features=n_input_channels*sample_length)


    def forward(self, X): # Input, X:128,30,sample_length
        n_input_channels = list(X.size())[1]
        sample_length = list(X.size())[-1]
        ########## Branch 1 ##########
        b1_X1 = self.b1_conv1(X)
        b1_X2 = self.b1_conv2(X)
        b1_X3 = self.b1_conv3(X)
        # Equation 7 in paper
        b1_att = self.b1_sigmoid(torch.bmm(torch.bmm(b1_X1,
                                                     torch.transpose(b1_X2, 1, 2))/math.sqrt(X.size(dim=-1)),
                                           b1_X3))
        b1_att = self.b1_bnorm1(b1_att+X)
        b1_att = torch.flatten(b1_att, start_dim=1)
        output1 = self.b1_linearchain(b1_att)
        b1_att = torch.unflatten(b1_att, 1, (n_input_channels, sample_length))
        output1 = torch.unflatten(output1, 1, (n_input_channels, sample_length))
        output1 = self.b1_bnorm2(b1_att+output1)

        ########## Branch 2 ##########
        b2_X1 = self.b2_conv1(X)
        b2_X2 = self.b2_conv2(X)
        b2_X3 = self.b2_conv3(X)
        X1_heads = torch.chunk(b2_X1, 5, dim=2)
        X2_heads = torch.chunk(b2_X2, 5, dim=2)
        X3_heads = torch.chunk(b2_X3, 5, dim=2)

        head_attn = [None]*5 # n_heads = 5
        for i in range(5): # n_heads = 5
            # Equation 8 in paper
            head_attn[i] = self.b2_sigmoid(torch.bmm(torch.bmm(X1_heads[i],
                                                         torch.transpose(X2_heads[i], 1, 2))/math.sqrt(X.size(dim=-1)),
                                               X3_heads[i]))
        b2_att = torch.cat(head_attn, 2)

        b2_att = self.b2_bnorm1(b2_att+X)
        b2_att = torch.flatten(b2_att, start_dim=1)
        output2 = self.b1_linearchain(b2_att)
        b2_att = torch.unflatten(b2_att, 1, (n_input_channels, sample_length))
        output2 = torch.unflatten(output2, 1, (n_input_channels, sample_length))
        output2 = self.b2_bnorm2(b2_att+output2)

        ################## Converge Branches ##################
        output = torch.cat((output1, output2), 1)
        output = torch.flatten(output, start_dim=1)
        output = self.converge(output)
        output = torch.unflatten(output, 1, (30,-1))
        return output
