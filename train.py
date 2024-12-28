from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
import copy
from Datasets.sleep_dataset import SleepDataset
def _train_and_validate_fold(model, loss_fn, train_loader, valid_loader, device):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3,
                                 weight_decay=1e-3,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    for epoch in range(100): # n_epochs = 100 following paper
        model.train()
        train_loss = 0.0
        for batch in enumerate(train_loader):  # x : torch.Size([128, 3000, 1]), y : torch.Size([128]):
            x, y = batch # ensure this gives tensor of specified size
            x, y = x.to(device), y.to(device)
            x = torch.permute(x, (0, 2, 1)) #Pytorch uses NCHW, we get x : torch.Size([128, 1, 3000])

            optimizer.zero_grad()
            yhat = model(x)
            loss = loss_fn(input=yhat, target=y) # input size = (minibatch, C), # target size = (C)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation for this fold after each epoch
        model.eval()
        valid_loss = 0.0
        valid_metrics = []
        with torch.no_grad():
            for batch_nr, (x, y) in enumerate(valid_loader):  # x : torch.Size([128, 3000, 1]), y : torch.Size([128]):
                x = torch.permute(x, (0, 2, 1))  # Pytorch uses NCHW, we get x : torch.Size([128, 1, 3000])
                yhat = model(x)
                loss = loss_fn(yhat, y)
                valid_loss += loss.item()

                # Calculate validation metrics (accuracy, F1 score, etc.)
                # TODO: apply max onto logits to get a prediction / class_label
                valid_metrics.append(calculate_metrics(yhat, y))

        # Log or print validation results
        print(f"Epoch {epoch}: Val Loss = {valid_loss/len(valid_loader)}")
        scheduler.step()

    return valid_metrics


def train(model : nn.Module, dataset : SleepDataset, n_kfolds : int=20, loss_fn=nn.CrossEntropyLoss()):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    except RuntimeError as e:
        print(f"Device error: {e}")
        # Fallback to CPU
        device = torch.device('cpu')
        model = model.to(device)


    kf = KFold(n_splits=n_kfolds, shuffle=False)  # We do not shuffle folds, but shuffle samples in each one instead

    folds_train_results = [] # each fold would give its own (loss, accuracy and...) curves
    folds_valid_results = [] # each fold would give its own (loss, accuracy and...) curves



    for fold_nr, (train_ids, test_ids) in enumerate(kf.split(dataset)):  # train/test_ids: numpy.ndarray of sample/segment indices

        # Initialize model for this fold (you might want to reset weights)
        #current_model = copy.deepcopy(model)

        train_loader = DataLoader(dataset=dataset,
                                  batch_size=128,
                                  sampler=torch.utils.data.SubsetRandomSampler(train_ids))  # sampler draws from subset randomly

        valid_loader = DataLoader(dataset=dataset,
                                 batch_size=128,
                                 sampler=torch.utils.data.SubsetRandomSampler(test_ids))  # sampler draws from subset randomly


        # model.attribute = list(model.attribute)  # where attribute was dict_keys
        # https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/25
        model_clone = copy.deepcopy(model)

        # Train and validate
        fold_train_results, fold_valid_results = _train_and_validate_fold(
            model=model_clone,
            loss_fn=loss_fn,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device)

        folds_train_results.append(fold_train_results)
        folds_valid_results.append(fold_valid_results)
    #averaged_train_results, averaged_train_results = process_results(folds_train_results, folds_valid_results) # Lists from fold are averaged
