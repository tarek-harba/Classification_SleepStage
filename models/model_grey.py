import torch.nn as nn
from modules import afr, mrcnn, tce
import torch


class ModelGrey(nn.Module):
    # MRCNN
    def __init__(self, shhs: bool):
        super().__init__()

        self.module1 = mrcnn.MRCNN(shhs=shhs)
        self.linear = nn.Linear(
            in_features=self.module1.output_n_channels * self.module1.output_sample_length,
            out_features=5)

        # softmax is added later, not to interfere with loss function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output1 = self.module1(x)
        output1 =  torch.flatten(output1, start_dim=1)
        yhat = self.linear(output1)
        return yhat
