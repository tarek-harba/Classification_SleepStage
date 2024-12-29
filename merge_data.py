import numpy as np
import os
import sys


def explore_sample_file(file_path: str) -> None:
    zipped_arr = np.load(file_path)
    print(sorted(zipped_arr.files))
    for file_name in zipped_arr.files:
        print(file_name, ", Size:", zipped_arr[file_name].size)


def merge_files(src_dir: str, output_name: str) -> None:
    x, y = [], []
    # ly = 0
    for subdir, dirs, files in os.walk(src_dir):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith(".npz"):
                zipped_arr = np.load(filepath)
                [x.append(segment) for segment in zipped_arr["x"]]
                [y.append(segment_label) for segment_label in zipped_arr["y"]]
                # ly+= zipped_arr['y'].size

    # print(ly)
    # print(len(x), len(y))
    data_dict = {"x": x, "y": y}
    np.savez(f"{output_name}.npz", **data_dict)
    print("Files merged and saved!")


if __name__ == "__main__":
    # explore_sample_file()
    src_dir = r"C:\Users\Doppler\PycharmProjects\ML_Paper_SleepClassification\preprocess\Final Downloaded Data\Preprocessed SHHS1 dataset - npz"
    output_name = r"C:\Users\Doppler\PycharmProjects\ML_Paper_SleepClassification\Datasets\data_shhs1"
    merge_files(src_dir, output_name)
