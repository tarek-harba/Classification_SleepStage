import numpy as np
import matplotlib.pyplot as plt
import random


def plot_samples(data_path_dict: dict) -> None:
    """
    Randomly selects a 30-second segment from each dataset specified in the input dictionary,
    and plots these segments on separate subplots. Each plot represents a segment from a dataset with its corresponding label.

    Parameters:
    -----------
    data_path_dict : dict
        A dictionary where keys are dataset names ("edf20", "edf78", "shhs1") and values are the file paths to the corresponding `.npz` files.

    Returns:
    --------
    None
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axis = plt.subplots(3, sharex=False, figsize=(10, 8))
    plt_nr = 0

    for data_name, data_path in data_path_dict.items():
        zipped_arr = np.load(data_path)
        upper_bound = len(zipped_arr["y"]) - 1
        sample_idx = random.randint(0, upper_bound)

        sample_segment_values = np.ndarray.flatten(zipped_arr["x"][sample_idx])
        sample_segment_label = zipped_arr["y"][sample_idx]

        axis[plt_nr].plot(
            sample_segment_values,
            label=f"Segment Label: {sample_segment_label}",
            linewidth=1.5,
        )
        axis[plt_nr].set_xlim(0, len(sample_segment_values) - 1)
        axis[plt_nr].legend(loc="upper right", fontsize=10, frameon=True, shadow=True)
        axis[plt_nr].set_title(f"Dataset: {data_name}", fontsize=12, fontweight="bold")
        axis[plt_nr].set_xlabel("Sample Number", fontsize=10)
        axis[plt_nr].tick_params(axis="both", labelsize=9)
        plt_nr += 1

    fig.text(
        0.04,
        0.5,
        "Channel-Recording Voltage",
        va="center",
        rotation="vertical",
        fontsize=12,
        fontweight="bold",
    )

    fig.suptitle(
        "Random 30-second Segments from each Dataset", fontsize=16, fontweight="bold"
    )

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.94])
    plt.show()


if __name__ == "__main__":
    data_path_dict = {
        "edf20": r"C:\Users\Doppler\PycharmProjects\ML_Paper_SleepClassification\Datasets\data_edf_20.npz",
        "edf78": r"C:\Users\Doppler\PycharmProjects\ML_Paper_SleepClassification\Datasets\data_edf_78.npz",
        "shhs1": r"C:\Users\Doppler\PycharmProjects\ML_Paper_SleepClassification\Datasets\data_shhs1.npz",
    }
    plot_samples(data_path_dict)
