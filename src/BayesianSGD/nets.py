""" Torch implementation of neural network architectures for MNIST and CIFAR datasets """

import torch
import torch.nn.functional as F
import numpy
import typing


class torch_net(torch.nn.Module):

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def __init__(self, dataset_name, input_shape, data_format):
        super(torch_net, self).__init__()
        self.dataset_name = dataset_name
        self.input_shape = input_shape
        self.data_format = data_format

        in_channels = None
        if data_format == "channels_last":
            in_channels = input_shape[2]
        elif data_format == "channels_first":
            in_channels = input_shape[0]

        if dataset_name == "mnist":
            self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3)
            self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
            self.pool = torch.nn.MaxPool2d((2, 2))
            self.drop1 = torch.nn.Dropout2d(p=0.25)
            self.fc1 = torch.nn.Linear(in_features=64*24*24, out_features=120)
            self.drop2 = torch.nn.Dropout2d(p=0.5)
            self.fc2 = torch.nn.Linear(in_features=120, out_features=10)
        elif dataset_name == "cifar":
            raise NotImplementedError

    def forward(self, x):
        if self.data_format == "channels_last":
            if type(x) is numpy.ndarray:
                x = numpy.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                x = numpy.rollaxis(x, 3, 1)
            else:
                x = x.permute(0,3,2,1)

        if self.dataset_name == "mnist":
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1) # flatten all except batch dimension
            x = F.relu(self.fc1(x))
            x = F.softmax(self.fc2(x), dim=1) # todo: dim deve essere 0 o 1?
        elif self.dataset_name == "cifar":
            raise NotImplementedError

        return x