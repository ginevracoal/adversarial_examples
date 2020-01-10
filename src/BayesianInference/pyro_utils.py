from utils import load_dataset
from torch.utils.data import DataLoader


def data_loaders(dataset_name, batch_size, n_inputs):
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = \
        load_dataset(dataset_name=dataset_name, n_samples=n_inputs, test=False)

    train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=batch_size)
    test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=batch_size)

    return train_loader, test_loader, data_format, input_shape