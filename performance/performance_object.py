import torch
import torchmetrics
from performance.metrics import compute_performance_metrics


class PerformanceObject:
    """
    A utility class to compute and store performance metrics for multi-class classification
    across epochs and k-folds in a machine learning pipeline.

    Attributes:
        n_epochs (int): Number of epochs in the training process.
        n_kfolds (int): Number of folds in k-fold cross-validation.
        n_classes (int): Number of classes in the classification problem. Default is 5.
        micro_perf (list): Stores per-class micro-level metrics [precision, recall, f1_score, g_mean, tp]
            for each epoch, fold, and class.
        macro_perf (list): Stores macro-level metrics [loss, kappa, MF1, MGm, Accuracy] for each epoch and fold.
        cm (torch.Tensor): A confusion matrix of shape (n_kfolds, n_epochs, n_classes, n_classes) for each fold and epoch.
        confmat (torchmetrics.ConfusionMatrix): A PyTorch Metrics object for computing confusion matrices for multiclass classification.
        cohenkappa (torchmetrics.CohenKappa): A PyTorch Metrics object for computing Cohen's Kappa.
    """

    def __init__(self, n_epochs: int, n_kfolds: int, n_metrics=5, n_classes=5):
        """
        Initializes the performance object with the specified number of epochs, k-folds, metrics, and classes.

        Args:
            n_epochs (int): Number of epochs.
            n_kfolds (int): Number of k-folds.
            n_metrics (int): Number of metrics. Default is 5.
            n_classes (int): Number of classes. Default is 5.
        """
        self.n_epochs = n_epochs
        self.n_kfolds = n_kfolds
        self.n_classes = n_classes
        # micro: fold_list(epoch_list(class_list([precision, recall, f1_score, g_mean, tp])))
        self.micro_perf = [
            [
                [[float(0)] * n_metrics for c in range(self.n_classes)]
                for e in range(self.n_epochs)
            ]
            for k in range(self.n_kfolds)
        ]
        # macro: fold_list(epoch_list([loss, kappa, MF1, MGm, Accuracy]))
        self.macro_perf = [
            [[float(0)] * n_metrics for e in range(self.n_epochs)]
            for k in range(self.n_kfolds)
        ]
        self.cm = torch.zeros(
            self.n_kfolds, self.n_epochs, self.n_classes, self.n_classes
        )

        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=5)
        self.cohenkappa = torchmetrics.CohenKappa(task="multiclass", num_classes=5)

    def compute_micro_perf(self) -> None:
        """
        Calculates and populates the `micro_perf` attribute with precision, recall, f1-score,
        g-mean, and true positives for each class, fold, and epoch.
        """
        # micro: fold_list(epoch_list(class_list([precision, recall, f1_score, g_mean, tp])))
        # macro: fold_list(epoch_list([loss, kappa, MF1, MGm, Accuracy]))
        for epoch_nr in range(self.n_epochs):
            for fold_nr in range(self.n_kfolds):
                conf_matrix = self.cm[fold_nr][epoch_nr]
                precision, recall, f1_score, g_mean, tp = compute_performance_metrics(
                    conf_matrix
                )
                for class_nr in range(self.n_classes):
                    # [precision, recall, f1_score, g_mean, tp]
                    self.micro_perf[fold_nr][epoch_nr][class_nr][0] = float(
                        precision[class_nr]
                    )
                    self.micro_perf[fold_nr][epoch_nr][class_nr][1] = float(
                        recall[class_nr]
                    )
                    self.micro_perf[fold_nr][epoch_nr][class_nr][2] = float(
                        f1_score[class_nr]
                    )
                    self.micro_perf[fold_nr][epoch_nr][class_nr][3] = float(
                        g_mean[class_nr]
                    )
                    self.micro_perf[fold_nr][epoch_nr][class_nr][4] = float(
                        tp[class_nr]
                    )

    def compute_macro_perf(self) -> None:
        """
        Calculates and populates the `macro_perf` attribute with average loss, kappa, MF1, MGm,
        and accuracy for each fold and epoch.
        """
        # micro: fold_list(epoch_list(class_list([precision, recall, f1_score, g_mean, tp])))
        # macro: fold_list(epoch_list([loss, kappa, MF1, MGm, Accuracy]))
        for epoch_nr in range(self.n_epochs):
            for fold_nr in range(self.n_kfolds):
                micro: torch.Tensor = torch.Tensor(self.micro_perf)
                # average across classes
                class_avg_micro: torch.Tensor = torch.mean(
                    micro, dim=2
                )  # result.size() : n_epochs, n_folds, [loss, kappa, MF1, MGm, Accuracy]
                avg_micro: list = torch.Tensor.tolist(class_avg_micro)
                self.macro_perf[fold_nr][epoch_nr][-1] = avg_micro[fold_nr][epoch_nr][
                    -1
                ]  # Accuracy
                self.macro_perf[fold_nr][epoch_nr][-2] = avg_micro[fold_nr][epoch_nr][
                    -2
                ]  # MGm
                self.macro_perf[fold_nr][epoch_nr][-3] = avg_micro[fold_nr][epoch_nr][
                    -3
                ]  # MF1

    def log_epoch(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        loss: float,
        fold_nr: int,
        epoch_nr: int,
    ) -> None:
        """
        Logs the performance for a specific fold and epoch based on predictions and targets.

        Args:
            prediction (torch.Tensor): Model predictions for the epoch.
            target (torch.Tensor): Ground-truth labels for the epoch.
            loss (float): Loss value for the epoch.
            fold_nr (int): Current fold number.
            epoch_nr (int): Current epoch number.
        """
        self.cm[fold_nr][epoch_nr] = self.confmat(prediction, target)
        kappa = self.cohenkappa(prediction, target).item()
        self._log_kappa(fold_nr, epoch_nr, kappa)
        self._log_loss(fold_nr, epoch_nr, loss)

    def _log_kappa(self, fold_nr: int, epoch_nr: int, kappa: float) -> None:
        """
        Logs the Cohen's Kappa value for a specific fold and epoch.

        Args:
            fold_nr (int): Current fold number.
            epoch_nr (int): Current epoch number.
            kappa (float): Cohen's Kappa value.
        """
        self.macro_perf[fold_nr][epoch_nr][1] = kappa

    def _log_loss(self, fold_nr: int, epoch_nr: int, loss: float) -> None:
        """
        Logs the loss value for a specific fold and epoch.

        Args:
            fold_nr (int): Current fold number.
            epoch_nr (int): Current epoch number.
            loss (float): Loss value.
        """
        self.macro_perf[fold_nr][epoch_nr][0] = loss


# From prepare_physionet and authors' notes in the research paper we know that:
# class_labels = {
#     0: "W",
#     1: "N1",
#     2: "N2",
#     3: "N3",
#     4: "REM",
# }
