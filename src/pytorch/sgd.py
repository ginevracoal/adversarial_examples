"""
Torch implementation of basic SGD and approximate Bayesian Inference SGD from Mandt et al., 2017.
"""

from collections import defaultdict
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import numpy as np
from torch.autograd import grad
import torch
import copy


class SGD(Optimizer):
    """ Simplified version of torch.SGD without Nesterov momentum. """
    def __init__(self, params, lr, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SGD, self).__init__(params, defaults)

        self.lr = lr
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []

        # to ensure compatibility with Optimizer base class
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def step(self, *args, **kwargs):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        # to ensure compatibility with Optimizer base class
        loss = None
        # if closure is not None:
        #     loss = closure()

        # for group in self.param_groups:
        weights = self.param_groups[0]

        for layer_params in weights['params']:
            if layer_params.grad is None:
                continue
            gradient = layer_params.grad.data
            layer_params.data -= -self.lr * gradient
        return loss


class BayesianSGD(SGD):

    def __init__(self, params, loss_fn):
        super(BayesianSGD, self).__init__(params=params, lr=None)
        # todo: che learning rate uso al passo zero?
        self.loss_fn = loss_fn
        self.k = 1 # decaying learning rate
        self.noise_covariance = {}
        # noise covariance matrix approximation

    def update_noise_covariance(self, outputs, labels, layer_params, params, layer_idx):
        """
        Updates the approximation of the noise covariance matrix at current epoch, then computes its Cholesky
        decomposition.
        :param C_old: old noise covariance approximation, type=np.ndarray
        :param k: decaying learning rate for timestep t, type=int
        return
        :param C_new: new noise covariance approximation C_new, type=np.ndarray
        :param B: Cholesky decomposition factor B, type=np.ndarray
        """
        # todo: questi gradienti a quali pesi e loss si riferiscono adesso?

        # print(outputs[0], labels[0])
        # exit()
        loss_1 = self.loss_fn(outputs[0:1], labels[0:1])
        loss_1.backward(retain_graph=True)
        g1 = copy.deepcopy(torch.unsqueeze(torch.flatten(layer_params.grad.data), dim=0))
        # g1 = copy.deepcopy(torch.flatten(layer_params.grad.data, start_dim=1))
        print("g1: ",g1.shape)

        loss_S = self.loss_fn(outputs, labels)
        loss_S.backward(retain_graph=True)
        gS = copy.deepcopy(torch.unsqueeze(torch.flatten(layer_params.grad.data), dim=0))
        # gS = copy.deepcopy(torch.flatten(layer_params.grad.data, start_dim=1))

        k = 1/(params['epoch']+1)

        if params['epoch']==0: #C_old is None:
            C_new = torch.mm((g1 - gS).t(), (g1 - gS))
        else:
            C_old = self.noise_covariance[str(layer_idx)]
            C_new = (1-k) * C_old + k * torch.mm((g1 - gS).t(),(g1 - gS))

        self.noise_covariance.update({str(layer_idx): C_new})
        print(len(self.noise_covariance))
        print("C_new: ", C_new.shape)

        # print(torch.eig(C_new))
        # exit()
        # B = np.linalg.cholesky(C_new)
        # return B

        return C_new

    def update_learning_rate(self, layer_params, outputs, labels, optimizer_params, layer_idx):
        """
        Computes the optimal layer-wise learning rate at current time, for performing constant SGD on a minibatch of
        samples.
        """
        batch_size = len(outputs)
        n_layer_weights = np.prod([x for x in layer_params.shape])
        total_n_samples = optimizer_params['n_training_samples']
        C = self.update_noise_covariance(outputs=outputs, labels=labels, layer_params=layer_params,
                                         params=optimizer_params, layer_idx=layer_idx)
        trace = np.trace(C)

        optimal_lr = (2 * batch_size * n_layer_weights) / (total_n_samples * trace)
        return optimal_lr

    def step(self, outputs, labels, optimizer_params):
        loss = None
        weights = self.param_groups[0]

        for layer_idx, layer_params in enumerate(weights['params']):
            if layer_params.grad is None:
                continue
            gradient = layer_params.grad.data
            # print(optimizer_params)
            # exit()
            # self.C_old = None
            lr = self.update_learning_rate(layer_params=layer_params, outputs=outputs, labels=labels,
                                           optimizer_params=optimizer_params, layer_idx=layer_idx)
            layer_params.data -= -lr * gradient

        return loss