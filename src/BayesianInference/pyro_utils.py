import itertools

from utils import load_dataset
from torch.utils.data import DataLoader
import random
import numpy as np


def data_loaders(dataset_name, batch_size, n_inputs, shuffle=False):
    random.seed(0)
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = \
        load_dataset(dataset_name=dataset_name, n_samples=n_inputs, test=False)

    train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader, data_format, input_shape


def slice_data_loader(data_loader, slice_size):
    images_list = []
    labels_list = []
    count = 1
    for images, labels in data_loader:
        for idx in range(len(images)):
            if count > slice_size:
                break
            count += 1
            images_list.append(np.array(images[idx]))
            labels_list.append(np.array(labels[idx]))
    images = np.array(images_list)
    labels = np.array(labels_list)

    print(f"\nSliced data_loader shapes: images = {images.shape}, labels = {labels.shape}")
    data_loader_slice = DataLoader(dataset=list(zip(images, labels)), batch_size=128)
    return data_loader_slice
