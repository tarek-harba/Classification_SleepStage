from modules.mrcnn import MRCNN
from Datasets.sleep_dataset import SleepDataset
import torch.nn as nn
from train import train
from calculate_weight import calculate_weight

from sklearn.utils.class_weight import compute_class_weight
if __name__ == '__main__':
    edf20_path = r'C:\Users\Doppler\PycharmProjects\Classification_SleepStage\Datasets\data_edf_20.npz'
    edf20_dataset = SleepDataset(edf20_path)
    weight = calculate_weight(edf20_dataset)
    loss_fn = nn.CrossEntropyLoss(weight=weight,reduction='none') # Weight : Tensor of size `C` and floating point dtype
    model = MRCNN()
    train(model=model, dataset=edf20_dataset, n_kfolds=20, loss_fn=loss_fn)


