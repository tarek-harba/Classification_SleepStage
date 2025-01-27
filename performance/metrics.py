import torch


def compute_performance_metrics(conf_matrix: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Computes performance metrics for a multi-class classification task based on the confusion matrix.

    Args:
        conf_matrix (torch.Tensor): A confusion matrix of shape (n_classes, n_classes).

    Returns:
        tuple[torch.Tensor, ...]: A tuple containing the following metrics for each class:
            - precision (torch.Tensor): Precision.
            - recall (torch.Tensor): Recall (sensitivity).
            - f1_score (torch.Tensor): F1 score.
            - g_mean (torch.Tensor): Geometric mean of recall and specificity.
            - tp (torch.Tensor): True positive rate.
    """
    # Calculate basic metrics
    # Calculate TP_rate
    actual_pos_count = conf_matrix.sum(dim=1)
    actualpos_predpos_count = torch.diag(conf_matrix, diagonal=0)
    tp_rate = actualpos_predpos_count / actual_pos_count
    # Calculate FP_rate
    actual_neg_count = torch.Tensor(
        [actual_pos_count.sum() - actual_pos_count[i] for i in range(5)]
    )
    actualneg_predpos_count = conf_matrix.sum(dim=0) - torch.diag(
        conf_matrix, diagonal=0
    )
    fp_rate = actualneg_predpos_count / actual_neg_count
    # Calculate FN_rate
    actualpos_predneg_count = conf_matrix.sum(dim=1) - torch.diag(
        conf_matrix, diagonal=0
    )
    fn_rate = actualpos_predneg_count / actual_pos_count
    # Calculate TN_rate
    total_samples = conf_matrix.sum().sum()
    tn_count = (total_samples - actual_pos_count) - actualneg_predpos_count
    tn_rate = tn_count / actual_neg_count

    # Calculate advanced metrics
    eps = 1e-7  # used to avoid division by zero
    precision = tp_rate / (tp_rate + fp_rate + eps)
    recall = tp_rate / (tp_rate + fn_rate + eps)  # a.k.a. sensitivity
    specificity = tn_rate / (tn_rate + fp_rate + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    g_mean = torch.sqrt(recall * specificity)

    # class_metrics: [precision, recall, f1_score, g_mean, tp]
    return precision, recall, f1_score, g_mean, tp_rate
