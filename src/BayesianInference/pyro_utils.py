from utils import load_dataset
from torch.utils.data import DataLoader
import random

def data_loaders(dataset_name, batch_size, n_inputs, shuffle=False):
    random.seed(0)
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = \
        load_dataset(dataset_name=dataset_name, n_samples=n_inputs, test=False)

    train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader, data_format, input_shape



