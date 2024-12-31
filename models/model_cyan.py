import torch.nn as nn
from modules import afr, mrcnn, tce
import torch

class ModelCyan(nn.Module):
    # MRCNN+AFR+TCE
    def __init__(self, shhs: bool):
        super().__init__()

        self.module1 = mrcnn.MRCNN(shhs=shhs)
        self.module2 = afr.AFR(input_n_channels=self.module1.output_n_channels,
                               input_sample_length=self.module1.output_sample_length)
        self.module3 = tce.TCE(input_n_channels=self.module2.output_n_channels,
                               input_sample_length=self.module2.output_sample_length)

        self.linear = nn.Linear(
            in_features=self.module3.output_n_channels * self.module3.output_sample_length, out_features=5)

        # softmax is added later, not to interfere with loss function

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        output1 = self.module1(x)
        output2 = self.module2(output1)
        output3 = self.module3(output2)
        output3 =  torch.flatten(output3, start_dim=1)
        yhat = self.linear(output3)
        return yhat
