import torch.nn as nn
from modules import afr, mrcnn, tce


class ModelCyan(nn.Module):
    # MRCNN+AFR+TCE
    def __init__(self, shhs : bool):
        super().__init__()
        input_channels = 30
        if shhs is False:
            input_length = 80

        else:
            input_length = 99

        self.model = nn.Sequential(mrcnn.MRCNN(),
                              afr.AFR(),
                              tce.TCE(input_length=input_length))

        self.linear = nn.Linear(in_features=30*input_length,out_features=5,bias=False)
    def forward(self, x):
        output = self.model(x)
        output = self.linear(output)
        return output