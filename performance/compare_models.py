import torch, os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# macro_labels = ["loss", "kappa", "MF1", "MGm", "Accuracy"]
def _plot_graphs(models_perf: dict) -> None:
    """
    Plots performance metrics for multiple models in a 2x2 grid of bar charts.

    Args:
        models_perf (dict): A dictionary where keys represent model colors
                            (used as bar colors) and values are lists of
                            performance metrics [loss, kappa, MF1, MGM, Accuracy].

    Behavior:
        - Creates four bar graphs for Accuracy, Macro F1-score, Kappa, and Macro G-mean.
        - Each bar graph corresponds to a specific metric and displays performance for all models.
    """
    # Define the colors for the bars
    colors = list(models_perf.keys())
    labels = [
        "MRCNN",
        "MRCNN+AFR",
        "MRCNN+TCE",
        "MRCNN+AFR+TCE",
        "AttnSleep",
    ]  # Legend names

    # Extract metrics for each bar graph
    kappa = [value[1] for value in models_perf.values()]
    mf1 = [value[2] for value in models_perf.values()]
    mgm = [value[3] for value in models_perf.values()]
    acc = [value[4] for value in models_perf.values()]

    # Create subplots (2 rows x 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Labels for each subplot
    labels_below = ["Accuracy", "Macro F1-score", "Kappa", "Macro G-mean"]

    # Use seaborn style (without using the seaborn package)
    plt.style.use("seaborn-v0_8-darkgrid")

    # Plot the bar graphs (vertical bars)
    axes[0, 0].bar(range(len(acc)), acc, color=colors)
    axes[0, 1].bar(range(len(mf1)), mf1, color=colors)
    axes[1, 0].bar(range(len(kappa)), kappa, color=colors)
    axes[1, 1].bar(range(len(mgm)), mgm, color=colors)

    # Add dashed horizontal gridlines to all plots
    for ax in axes.flat:
        ax.grid(
            True, axis="y", linestyle="--", linewidth=0.7
        )  # Dashed horizontal lines (grid for y-axis)
        ax.set_xticks([])  # Remove x-axis tick labels

    # Assign the labels below the respective axes
    for ax, label in zip(axes.flat, labels_below):
        ax.set_xlabel(
            label, labelpad=15, fontsize=14
        )  # Added fontsize=14 for larger labels

    # Create custom legend with colored rectangles
    legend_handles = [
        Patch(color=color, label=label) for color, label in zip(colors, labels)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.9),
    )

    # Adjust layout to avoid overlap
    plt.tight_layout(rect=[0, 0.05, 1, 0.85])  # Increased bottom margin

    # Additional adjustment to ensure no overlap
    plt.subplots_adjust(hspace=0.4)  # Increase vertical space between subplots

    # Show the plot
    plt.show()


def compare_models(dir_path: str) -> None:
    """
    Compares models stored in a directory by loading their performance metrics and visualizing them.

    Args:
        dir_path (str): Path to the directory containing model files (.pth format).

    Behavior:
        - Scans the directory for `.pth` files.
        - Loads the performance metrics (macro-level) from each file.
        - Stores the relevant metrics (Accuracy, Macro F1-score, Kappa, Macro G-mean) for comparison.
        - Passes the metrics to `_plot_graphs` for visualization.
    """
    models_perf = dict.fromkeys(["grey", "blue", "orange", "cyan", "purple"])

    for filename in os.listdir(dir_path):
        if filename.endswith(".pth"):
            ######### Get model's results path and its color/name #########
            file_path = os.path.join(dir_path, filename)
            model_color = filename.split("_")[0]
            ######### Load that Model #########
            perf_results = torch.load(file_path, weights_only=False)
            valid_perf = perf_results["valid"]
            valid_macro = torch.Tensor(valid_perf["macro"])

            ######### Store Relevant Information #########
            valid_macro_avg = torch.mean(valid_macro, dim=0)
            valid_macro_avg = torch.Tensor.tolist(valid_macro_avg)
            model_results = valid_macro_avg[-1]
            models_perf[model_color] = model_results

    _plot_graphs(models_perf)


# macro_labels = ["loss", "kappa", "MF1", "MGm", "Accuracy"]
