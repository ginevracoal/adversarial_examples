from directories import *
import os
from torch import nn
import torch
import pyro
from pyro.distributions import OneHotCategorical, Normal
import torch.nn.functional as nnf


class NN(nn.Module):

    def __init__(self, input_size, hidden_size, n_classes):
        self.dim = -1
        super(NN, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.LeakyReLU(),
                                   nn.Linear(hidden_size, n_classes),
                                   nn.Softmax(dim=self.dim))

    def forward(self, inputs):
        return self.model(inputs)


class BNN(nn.Module):
    def __init__(self, input_size, device):
        super(BNN, self).__init__()
        self.device = device
        self.n_classes = 10
        self.hidden_size = 128
        self.input_size = input_size
        self.net = NN(input_size=input_size, hidden_size=self.hidden_size, n_classes=self.n_classes)


    def model(self, inputs, labels=None):
        batch_size = inputs.size(0)
        flat_inputs = inputs.to(self.device).view(-1, self.input_size)
        # first layer
        fc1w_mean = torch.zeros(self.input_size, self.hidden_size)
        fc1w_scale = torch.ones(self.input_size, self.hidden_size)
        fc1b_mean = torch.zeros(self.hidden_size)
        fc1b_scale = torch.ones(self.hidden_size)
        # second layer
        outw_mean = torch.zeros(self.hidden_size, self.n_classes)
        outw_scale = torch.ones(self.hidden_size, self.n_classes)
        outb_mean = torch.zeros(self.n_classes)
        outb_scale = torch.ones(self.n_classes)

        # sample priors
        fc1w_prior = pyro.sample('fc1w_prior', Normal(loc=fc1w_mean, scale=fc1w_scale))#.independent(2))
        fc1b_prior = pyro.sample('fc1b_prior', Normal(loc=fc1b_mean, scale=fc1b_scale))#.independent(1))
        outw_prior = pyro.sample('outw_prior', Normal(loc=outw_mean, scale=outw_scale))#.independent(2))
        outb_prior = pyro.sample('outb_prior', Normal(loc=outb_mean, scale=outb_scale))#.independent(1))

        # condition on the observed data
        # print(flat_inputs[0][345:400])
        out = nnf.leaky_relu(torch.matmul(flat_inputs,fc1w_prior) + fc1b_prior)
        logits = nnf.softmax(torch.matmul(out,outw_prior) + outb_prior, dim=self.net.dim)
        # print("\ncheck prob distributions:", logits.sum(1))
        cond_model = pyro.sample("obs", OneHotCategorical(logits=logits), obs=labels)
        # print("\nlogits.shape =",logits.shape)
        # print("\nlogits[0] =",logits[0])
        # print(cond_model)
        # exit()
        return cond_model

