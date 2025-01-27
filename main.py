from Data_Handling.sleep_dataset import SleepDataset
import torch.nn as nn
from train import train
from calculate_weight import calculate_weight
from models import model_cyan
from performance.process_results import *
import os

if __name__ == "__main__":
    ########################### Import data #############################
    edf_path = os.path.join(os.path.dirname(__file__), "Datasets", "data_edf_20.npz")
    ds = SleepDataset(edf_path)

    ########################### set Loss function and Weights (if weighted cross entropy) #############################
    # when using standard multi-class entropy, set weight and reduction to none, we are doing our own reduction in train
    #  by a simple mean of loss
    weight = calculate_weight(ds)
    loss_fn = nn.CrossEntropyLoss(
        weight=weight, reduction="none"  # always have reduction='none'
    )  # Weight : Tensor of size `C` and floating point dtype

    ########################### Select Model and train #############################
    model = model_cyan.ModelCyan(shhs=False)
    train_po, valid_po = train(
        model=model, dataset=ds, n_kfolds=20, n_epochs=40, loss_fn=loss_fn
    )
    file_path = save_model_results(
        model_name="purple", train_po=train_po, valid_po=valid_po
    )
    print("All Done! Finished Training + Saved performance metrics")
