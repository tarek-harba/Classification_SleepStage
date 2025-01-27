from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
import copy
import torch.nn.functional as F
import numpy as np
from performance.performance_object import PerformanceObject

from Data_Handling.sleep_dataset import SleepDataset


def _train_and_validate_fold(
    model,
    loss_fn,
    train_loader,
    valid_loader,
    device,
    n_epochs: int,
    fold_nr: int,
    train_po: PerformanceObject,
    valid_po: PerformanceObject,
) -> tuple[PerformanceObject, PerformanceObject]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=True,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10], gamma=0.1
    )

    softmax_operator = nn.Softmax(dim=1)  # for performance measure

    for epoch_nr in range(n_epochs):

        train_targets_list, train_predictions_list = [], []
        valid_targets_list, valid_predictions_list = [], []

        model.train()
        train_loss = 0.0
        for batch_index, (
            x,
            y,
        ) in enumerate(  # x : torch.Size([128, 3000, 1]), y : torch.Size([128])
            train_loader
        ):
            x, y = x.to(device=device), y.to(dtype=torch.long, device=device)
            x = torch.permute(
                x, (0, 2, 1)
            )  # Since Pytorch uses NCHW, we want x : torch.Size([128, 1, 3000])

            optimizer.zero_grad()
            yhat = model(x)
            t_loss = loss_fn(
                input=yhat, target=y
            )  # input size = (minibatch, C), # target size = (C)
            t_loss = torch.mean(t_loss)  # this yields loss per sample in a given batch
            t_loss.backward()
            optimizer.step()

            # log training results
            train_loss += t_loss.item() / 128
            train_targets_list.extend(
                y.tolist()
            )  # list of tensors, each can be seen as list of targets
            yhat_prob = softmax_operator(yhat)
            yhat_onehot = F.one_hot(torch.argmax(yhat_prob, dim=1), num_classes=5)
            train_predictions_list.extend(
                yhat_onehot.tolist()
            )  # list of tensors, each can be seen as list of targets

        # Validation for this fold after each epoch
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_nr, (x, y) in enumerate(
                valid_loader
            ):  # x : torch.Size([128, 3000, 1]), y : torch.Size([128]):
                x, y = x.to(device=device), y.to(dtype=torch.long, device=device)
                x = torch.permute(
                    x, (0, 2, 1)
                )  # Pytorch uses NCHW, we get x : torch.Size([128, 1, 3000])
                yhat = model(x)
                v_loss = loss_fn(
                    input=yhat, target=y
                )  # input size = (minibatch, C), # target size = (C)
                v_loss = torch.mean(v_loss)  # this yields loss per sample
                valid_loss += v_loss.item() / 128

                # Calculate validation metrics (accuracy, F1 score, etc.)

                valid_targets_list.extend(
                    y.tolist()
                )  # list of tensors, each can be seen as list of targets
                yhat_prob = softmax_operator(yhat)
                yhat_onehot = F.one_hot(torch.argmax(yhat_prob, dim=1), num_classes=5)
                valid_predictions_list.extend(
                    yhat_onehot.tolist()
                )  # list of tensors, each can be seen as list of targets

        # Log/print validation results
        train_po.log_epoch(
            prediction=torch.Tensor(train_predictions_list),
            target=torch.Tensor(train_targets_list),
            loss=train_loss,
            fold_nr=fold_nr,
            epoch_nr=epoch_nr,
        )
        valid_po.log_epoch(
            prediction=torch.Tensor(valid_predictions_list),
            target=torch.Tensor(valid_targets_list),
            loss=valid_loss,
            fold_nr=fold_nr,
            epoch_nr=epoch_nr,
        )
        print(
            f"Fold_nr:{fold_nr} Epoch_nr: {epoch_nr}: Train Loss = {train_loss}, Val Loss = {valid_loss}"
        )

        scheduler.step()

    return train_po, valid_po


def train(
    model: nn.Module,
    dataset: SleepDataset,
    loss_fn=nn.CrossEntropyLoss(),
    n_kfolds: int = 20,
    n_epochs: int = 100,
) -> tuple[PerformanceObject, PerformanceObject]:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    except RuntimeError as e:
        print(f"Device error: {e}")
        # Fallback to CPU
        device = torch.device("cpu")
        model = model.to(device)
    print(device)
    kf = KFold(
        n_splits=n_kfolds, shuffle=False
    )  # We do not shuffle folds, but shuffle samples in each fold instead

    #
    train_po = PerformanceObject(n_epochs=n_epochs, n_kfolds=n_kfolds)
    valid_po = PerformanceObject(n_epochs=n_epochs, n_kfolds=n_kfolds)

    for fold_nr, (train_ids, test_ids) in enumerate(
        kf.split(dataset)
    ):  # train/test_ids: numpy.ndarray of sample/segment indices
        # Initialize model for this fold (you might want to reset weights)
        # current_model = copy.deepcopy(model)
        # print("Overlap Status: ", any(np.isin(train_ids,test_ids)))
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=128,
            sampler=torch.utils.data.SubsetRandomSampler(train_ids),
        )  # sampler draws from subset randomly

        valid_loader = DataLoader(
            dataset=dataset,
            batch_size=128,
            sampler=torch.utils.data.SubsetRandomSampler(test_ids),
        )  # sampler draws from subset randomly

        # model.attribute = list(model.attribute)  # where attribute was dict_keys
        # https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/25
        model_clone = copy.deepcopy(model)
        # Train and validate
        train_po, valid_po = _train_and_validate_fold(
            model=model_clone,
            loss_fn=loss_fn,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            n_epochs=n_epochs,
            fold_nr=fold_nr,
            train_po=train_po,
            valid_po=valid_po,
        )
    train_po.compute_micro_perf()
    valid_po.compute_micro_perf()

    train_po.compute_macro_perf()
    valid_po.compute_macro_perf()

    return train_po, valid_po
