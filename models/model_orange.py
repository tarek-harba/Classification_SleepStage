import torch.nn as nn
from modules import afr, mrcnn, tce


class ModelOrange(nn.Module):
    # MRCNN+TCE
    def __init__(self, shhs: bool):
        super().__init__()
        n_input_channels = 30
        if shhs is False:
            sample_length = 80

        else:
            sample_length = 99

        self.model = nn.Sequential(
            mrcnn.MRCNN(),
            afr.AFR(),
            tce.TCE(n_input_channels=n_input_channels, sample_length=sample_length),
        )

        self.linear = nn.Linear(
            in_features=sample_length * n_input_channels, out_features=5, bias=False
        )

    def forward(self, x):
        output = self.model(x)
        output = self.linear(output)
        return output
