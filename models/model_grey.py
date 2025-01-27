import torch.nn as nn
from modules import afr, mrcnn, tce
import torch


class ModelGrey(nn.Module):
    # MRCNN
    """
    ModelCyan: A neural network model utilizing MRCNN module as described in authors' paper.

    Attributes:
        module1 (nn.Module): Instance of the `MRCNN` class.
        linear (nn.Linear): Fully connected layer for classification.

    Args:
        shhs (bool): A flag passed to the `MRCNN` module indicating whether to use
                     specific configurations for the SHHS dataset.
    """

    def __init__(self, shhs: bool):
        super().__init__()

        self.module1 = mrcnn.MRCNN(shhs=shhs)
        self.linear = nn.Linear(
            in_features=self.module1.output_n_channels
            * self.module1.output_sample_length,
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
        output1 = torch.flatten(output1, start_dim=1)
        yhat = self.linear(output1)
        return yhat
