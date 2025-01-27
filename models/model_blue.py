import torch.nn as nn
from modules import afr, mrcnn, tce
import torch


class ModelBlue(nn.Module):
    # MRCNN+AFR
    """
    ModelBlue: A neural network model combining MRCNN and AFR modules as described in authors' paper.

    Attributes:
        module1 (nn.Module): Instance of the `MRCNN` class.
        module2 (nn.Module): Instance of the `AFR` class.
        linear (nn.Linear): Fully connected layer for classification.

    Args:
        shhs (bool): A flag passed to the `MRCNN` module indicating whether to use
                     specific configurations for the SHHS dataset.
    """

    def __init__(self, shhs: bool):
        super().__init__()

        self.module1 = mrcnn.MRCNN(shhs=shhs)
        self.module2 = afr.AFR(
            input_n_channels=self.module1.output_n_channels,
            input_sample_length=self.module1.output_sample_length,
        )

        self.linear = nn.Linear(
            in_features=self.module2.output_n_channels
            * self.module2.output_sample_length,
            out_features=5,
        )

        # softmax is added later, not to interfere with loss function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, input_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 5), containing raw logits
                          for each of the 5 output classes.
        """
        output1 = self.module1(x)
        output2 = self.module2(output1)
        output2 = torch.flatten(output2, start_dim=1)
        yhat = self.linear(output2)
        return yhat
