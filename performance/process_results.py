import torch, os
from performance.performance_object import PerformanceObject
import matplotlib.pyplot as plt


def save_model_results(
    model_name: str, train_po: PerformanceObject, valid_po: PerformanceObject
) -> str:
    """
    Save the model performance results (confusion matrix, micro performance, macro performance)
    for training/validation to a file.

    Parameters:
    model_name (str): The name of the model, used to generate the file name.
    train_po (PerformanceObject): The performance object containing training metrics.
    valid_po (PerformanceObject): The performance object containing validation metrics.

    Returns:
    str: The file path where the performance results are saved.
    """
    train_perf = {
        "cm": train_po.cm,
        "micro": train_po.micro_perf,
        "macro": train_po.macro_perf,
    }

    valid_perf = {
        "cm": valid_po.cm,
        "micro": valid_po.micro_perf,
        "macro": valid_po.macro_perf,
    }

    perf_results = {"train": train_perf, "valid": valid_perf}

    file_path = os.path.join(
        os.path.dirname(__file__), model_name + "_performance_dict.pth"
    )
    torch.save(perf_results, file_path)
    return file_path


def _plot_results(
    train_epochs_loss: list,
    train_epochs_acc: list,
    valid_epochs_loss: list,
    valid_epochs_acc: list,
) -> None:
    """
    Plot the training and validation loss/accuracy across epochs.

    Parameters:
    train_epochs_loss (list): A list containing the training loss per epoch.
    train_epochs_acc (list): A list containing the training accuracy per epoch.
    valid_epochs_loss (list): A list containing the validation loss per epoch.
    valid_epochs_acc (list): A list containing the validation accuracy per epoch.

    Returns:
    None: Displays the plot of training and validation loss/accuracy.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot training/validation loss
    axes[0].plot(
        [i + 1 for i in range(len(train_epochs_loss))],
        train_epochs_loss,
        label="Training Loss",
        color="blue",
    )
    axes[0].plot(
        [i + 1 for i in range(len(valid_epochs_loss))],
        valid_epochs_loss,
        label="Validation Loss",
        color="orange",
    )
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch_nr")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Plot training/validation accuracy
    axes[1].plot(
        [i + 1 for i in range(len(train_epochs_acc))],
        train_epochs_acc,
        label="Training Accuracy",
        color="blue",
    )
    axes[1].plot(
        [i + 1 for i in range(len(valid_epochs_acc))],
        valid_epochs_acc,
        label="Validation Accuracy",
        color="orange",
    )
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    # Add a title for the whole figure
    fig.suptitle("Training/Validation Loss/Accuracy across epochs", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])


def process_results(file_path: str) -> None:
    """
    Process and display the saved model performance results including confusion matrices,
    micro results, and macro results (including loss and accuracy).

    Parameters:
    file_path (str): The path to the file containing the saved performance results.

    Returns:
    None: Displays the confusion matrix, micro performance results, and macro performance results
          along with plots for loss and accuracy across epochs.
    """
    perf_results = torch.load(file_path, weights_only=False)

    train_perf, valid_perf = perf_results["train"], perf_results["valid"]
    train_cm, train_micro, train_macro = (
        torch.Tensor(train_perf["cm"]),
        torch.Tensor(train_perf["micro"]),
        torch.Tensor(train_perf["macro"]),
    )
    valid_cm, valid_micro, valid_macro = (
        torch.Tensor(valid_perf["cm"]),
        torch.Tensor(valid_perf["micro"]),
        torch.Tensor(valid_perf["macro"]),
    )

    # TODO####################### Display Confusion Matrix Results ########################
    # Average across folds
    train_cm_fold_avg = torch.mean(train_cm, dim=0)

    # Display the result
    labels = ["W", "N1", "N2", "N3", "REM"]

    print("--" * 99)
    print("Training Confusion Matrix:")

    # Print the header row with proper alignment
    print(f"{'':8}" + " ".join(f"{label:8}" for label in labels))

    # Print the matrix with row labels and aligned columns
    for label, row in zip(
        labels, train_cm_fold_avg[-1]
    ):  # Access the last epoch's 5x5 sample
        print(f"{label:8}" + " ".join(f"{item:8.2f}" for item in row))

    print("--" * 99)

    # Average across folds
    valid_cm_fold_avg = torch.mean(valid_cm, dim=0)

    # Display the result
    labels = ["W", "N1", "N2", "N3", "REM"]

    print("--" * 99)
    print("Validation Confusion Matrix:")

    # Print the header row with proper alignment
    print(f"{'':8}" + " ".join(f"{label:8}" for label in labels))

    # Print the matrix with row labels and aligned columns
    for label, row in zip(
        labels, valid_cm_fold_avg[-1]
    ):  # Access the last epoch's 5x5 sample
        print(f"{label:8}" + " ".join(f"{item:8.2f}" for item in row))

    print("--" * 99)

    # TODO####################### Display Micro Results ########################
    train_micro_avg = torch.mean(train_micro, dim=0)
    train_micro_avg = torch.Tensor.tolist(train_micro_avg)
    # Column and row labels
    metric_labels = ["precision", "recall", "f1_score", "g_mean", "tp"]
    class_labels = ["W", "N1", "N2", "N3", "REM"]

    print("--" * 99)
    print("Training Micro Results:")

    # Print the header row with metric labels
    print(f"{'':8}" + "   ".join(f"{label:10}" for label in metric_labels))

    # Print each row with class labels
    for class_label, row in zip(
        class_labels, train_micro_avg[-1]
    ):  # Access the last epoch's 5x5 sample
        print(f"{class_label:8}" + " ".join(f"{item*100:10.2f}" for item in row))

    print("--" * 99)

    valid_micro_avg = torch.mean(valid_micro, dim=0)
    valid_micro_avg = torch.Tensor.tolist(valid_micro_avg)
    # Column and row labels
    metric_labels = ["precision", "recall", "f1_score", "g_mean", "tp"]
    class_labels = ["W", "N1", "N2", "N3", "REM"]

    print("--" * 99)
    print("Valid Micro Results:")

    # Print the header row with metric labels
    print(f"{'':8}" + "   ".join(f"{label:10}" for label in metric_labels))

    # Print each row with class labels
    for class_label, row in zip(
        class_labels, valid_micro_avg[-1]
    ):  # Access the last epoch's 5x5 sample
        print(f"{class_label:8}" + " ".join(f"{item*100:10.2f}" for item in row))

    print("--" * 99)

    # TODO####################### Plot Accuracy/Loss across Epochs from Macro ########################
    # Get Relevant Training Results
    # macro: fold_list(epoch_list([loss, kappa, MF1, MGm, Accuracy]))

    train_macro_avg = torch.mean(train_macro, dim=0)
    train_macro_avg = torch.Tensor.tolist(train_macro_avg)

    train_epochs_loss = [epoch_list[0] for epoch_list in train_macro_avg]
    train_epochs_acc = [epoch_list[-1] for epoch_list in train_macro_avg]

    # Get Relevant Validation Results
    valid_macro_avg = torch.mean(valid_macro, dim=0)
    valid_macro_avg = torch.Tensor.tolist(valid_macro_avg)

    valid_epochs_loss = [epoch_list[0] for epoch_list in valid_macro_avg]
    valid_epochs_acc = [epoch_list[-1] for epoch_list in valid_macro_avg]
    _plot_results(
        train_epochs_loss, train_epochs_acc, valid_epochs_loss, valid_epochs_acc
    )

    # TODO####################### Display Macro Results ########################
    macro_labels = ["loss", "kappa", "MF1", "MGm", "Accuracy"]

    print("--" * 99)
    print("Training Macro Results:")
    # Print training results with labels
    for label, value in zip(macro_labels, train_macro_avg[-1]):  # last epoch
        print(f"{label}: {value:.4f}")

    print("--" * 99)
    print("Validation Macro Results:")
    # Print validation results with labels
    for label, value in zip(macro_labels, valid_macro_avg[-1]):  # last epoch
        print(f"{label}: {value:.4f}")
    print("--" * 99)

    # Show the plot
    plt.show()


# micro: fold_list(epoch_list(class_list([precision, recall, f1_score, g_mean, tp])))
# macro: fold_list(epoch_list([loss, kappa, MF1, MGm, Accuracy]))
"""
    Storage Structure:
        - models_dict --> train/valid_dict --> mc/micro/macro_dict
-----------------------------------------------------------------------
- mc:    (1) avg across folds and print last epoch.
-----------------------------------------------------------------------
- micro: (1) avg across folds, epoch_nr[-1]: return all but tp
-----------------------------------------------------------------------
- macro: (1) avg across folds and plot loss / accuracy across epochs
         (2) avg across folds, epoch[-1]: return all but loss
"""
