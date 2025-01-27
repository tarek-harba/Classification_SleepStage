from Data_Handling.sleep_dataset import SleepDataset
import numpy as np
import torch

# From prepare_physionet and authors' notes in the research paper we know that:
# class_labels = {
#     0: "W",
#     1: "N1",
#     2: "N2",
#     3: "N3",
#     4: "REM",
# }


def calculate_weight(dataset: SleepDataset):
    """
    Calculates the class weights based on the dataset labels to address class imbalance.
    The weights are calculated using the formula provided in the supplementary materials of the research paper.

    Parameters:
    -----------
    dataset : SleepDataset
        A SleepDataset object

    Returns:
    --------
    torch.Tensor
        A tensor containing the calculated class weights, which can be used for loss function adjustments
        to handle class imbalance during model training.
    """
    y = list(dataset._y)
    m, k = len(y), len(list(set(y)))  # m = n_samples, k = n_classes

    mk = [
        y.count(0),
        y.count(1),
        y.count(2),
        y.count(3),
        y.count(4),
    ]  # counts of each class
    a, b, c = 1, 1.5, 2  # specified in paper's supplementary material

    mu = [b / k, c / k, b / k, a / k, b / k]  # {0:b/K, 1:c/K, 2:b/K, 3:a/K, 4:b/K}
    params = [np.max([1, np.log((mu[i] * m) / mk[i])]) for i in range(len(mu))]

    weight = np.multiply(mu, params)
    weight = torch.Tensor(weight)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except RuntimeError as e:
        print(f"Device error: {e}")
        # Fallback to CPU
        device = torch.device("cpu")
    weight = weight.to(device)
    return weight
