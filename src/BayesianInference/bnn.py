
from directories import *
import os
import torch.nn.functional as nnf
from torch.distributions import constraints
from torch import nn
import torch
import pyro
from pyro.contrib import bnn
from BayesianSGD.nets import torch_net
from utils import load_dataset, onehot_to_labels
from torch.distributions import OneHotCategorical
from torch.utils.data import DataLoader


def data_loaders(dataset_name, batch_size, n_samples):
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = \
        load_dataset(dataset_name=dataset_name, n_samples=n_samples, test=False)
    y_train = onehot_to_labels(y_train)
    y_test = onehot_to_labels(y_test)

    train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=batch_size)
    test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=batch_size)

    return train_loader, test_loader, data_format, input_shape


class BNN(nn.Module):
    def __init__(self, dataset_name, input_shape, data_format):
        super(BNN, self).__init__()
        self.net = torch_net(dataset_name=dataset_name, input_shape=input_shape, data_format=data_format)
        self.n_hidden = 1024
        self.n_classes = 10

    def model(self, inputs, labels=None, kl_factor=1.0):
        size = inputs.size(0)
        flat_inputs = inputs.view(-1, 784)
        # Set-up parameters for the distribution of weights for each layer `a<n>`
        a1_mean = torch.zeros(784, self.n_hidden)
        a1_scale = torch.ones(784, self.n_hidden)
        a1_dropout = torch.tensor(0.25)
        a2_mean = torch.zeros(self.n_hidden + 1, self.n_classes)
        a2_scale = torch.ones(self.n_hidden + 1, self.n_hidden)
        a2_dropout = torch.tensor(1.0)
        a3_mean = torch.zeros(self.n_hidden + 1, self.n_classes)
        a3_scale = torch.ones(self.n_hidden + 1, self.n_hidden)
        a3_dropout = torch.tensor(1.0)
        a4_mean = torch.zeros(self.n_hidden + 1, self.n_classes)
        a4_scale = torch.ones(self.n_hidden + 1, self.n_classes)
        with pyro.plate('data', size=size):
            # sample conditionally independent hidden layers
            h1 = pyro.sample('h1', bnn.HiddenLayer(flat_inputs, a1_mean, a1_dropout*a1_scale, non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_dropout*a2_scale, non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_dropout*a3_scale, non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: nnf.log_softmax(x, dim=-1),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))
            # One-hot encode labels
            labels = nnf.one_hot(labels) if labels is not None else None

            # Condition on the observed labels
            cond_model = pyro.sample('label', OneHotCategorical(logits=logits), obs=labels)
            return cond_model

    def guide(self, inputs, labels=None, kl_factor=1.0):
        size = inputs.size(0)
        flat_inputs = inputs.view(-1, 784)
        a1_mean = pyro.param('a1_mean', 0.01 * torch.randn(784, self.n_hidden))
        a1_scale = pyro.param('a1_scale', 0.1 * torch.ones(784, self.n_hidden),
                              constraint=constraints.greater_than(0.01))
        a1_dropout = pyro.param('a1_dropout', torch.tensor(0.25),
                                constraint=constraints.interval(0.1, 1.0))
        a2_mean = pyro.param('a2_mean', 0.01 * torch.randn(self.n_hidden + 1, self.n_hidden))
        a2_scale = pyro.param('a2_scale', 0.1 * torch.ones(self.n_hidden + 1, self.n_hidden),
                              constraint=constraints.greater_than(0.01))
        a2_dropout = pyro.param('a2_dropout', torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0))
        a3_mean = pyro.param('a3_mean', 0.01 * torch.randn(self.n_hidden + 1, self.n_hidden))
        a3_scale = pyro.param('a3_scale', 0.1 * torch.ones(self.n_hidden + 1, self.n_hidden),
                              constraint=constraints.greater_than(0.01))
        a3_dropout = pyro.param('a3_dropout', torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0))
        a4_mean = pyro.param('a4_mean', 0.01 * torch.randn(self.n_hidden + 1, self.n_classes))
        a4_scale = pyro.param('a4_scale', 0.1 * torch.ones(self.n_hidden + 1, self.n_classes),
                              constraint=constraints.greater_than(0.01))

        with pyro.plate('data', size=size):
            h1 = pyro.sample('h1', bnn.HiddenLayer(flat_inputs, a1_mean, a1_dropout*a1_scale, non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_dropout*a2_scale, non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_dropout*a3_scale, non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: nnf.log_softmax(x, dim=-1),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))

    def infer_parameters(self, dataloader, device, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, filename, relative_path=RESULTS+"bnn/"):
        filepath = relative_path + filename + ".pr"
        os.makedirs(os.path.dirname(relative_path), exist_ok=True)
        print("\nSaving params: ", filepath)
        pyro.get_param_store().save(filepath)

    def load(self, filename, relative_path=RESULTS+"bnn/"):
        filepath = relative_path+filename+".pr"
        print("\nLoading params: ", filepath)
        pyro.get_param_store().load(filepath)

    def evaluate_test(self, test_loader, device):
        self.eval()
        total = 0.0
        correct = 0.0
        for images, labels in test_loader:
            pred = self.forward(images.to(device).view(-1, 784), n_samples=1)
            total += labels.size(0)
            correct += (pred.argmax(-1) == labels.to(device)).sum().item()
        print(f"\nTest accuracy: {correct / total * 100:.5f}")
