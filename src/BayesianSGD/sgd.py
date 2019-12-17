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
import cProfile

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
        # loss = None
        # if closure is not None:
        #     loss = closure()

        # for group in self.param_groups:
        weights = self.param_groups[0]

        for layer_params in weights['params']:
            if layer_params.grad is None:
                continue
            gradient = layer_params.grad.data
            layer_params.data -= -self.lr * gradient
        # return loss


class BayesianSGD(SGD):

    def __init__(self, params, loss_fn):
        super(BayesianSGD, self).__init__(params=params, lr=None)
        # todo: che learning rate uso al passo zero?
        self.loss_fn = loss_fn
        self.k = 1 # decaying learning rate
        self.noise_covariance_traces = {}
        # noise covariance matrix approximation

    def update_traces(self, outputs, labels, layer_params, params, layer_idx):
        """
        Updates the approximation of the noise covariance matrix at current epoch, then computes its Cholesky
        decomposition.
        :param C_old: old noise covariance approximation, type=np.ndarray
        :param k: decaying learning rate for timestep t, type=int
        return
        :param C_new: new noise covariance approximation C_new, type=np.ndarray
        :param B: Cholesky decomposition factor B, type=np.ndarray
        """
        # grad, _ = torch.autograd.grad(params['loss1'], params['weights'].requires_grad_(True))

        # loss1 = nn.CrossEntropyLoss()(outputs[0:1], labels[0:1])
        # loss1.backward(retain_graph=True)
        # g1 = copy.deepcopy(torch.unsqueeze(torch.flatten(layer_params.grad.data), dim=0))
        # lossS = nn.CrossEntropyLoss()(outputs, labels)
        # lossS.backward()
        # gS = copy.deepcopy(torch.unsqueeze(torch.flatten(layer_params.grad.data), dim=0))

        # g1 = torch.unsqueeze(torch.flatten(params['w1'][layer_idx].grad.data), dim=0)
        # gS = torch.unsqueeze(torch.flatten(params['wS'][layer_idx].grad.data), dim=0)

        g1 = torch.unsqueeze(torch.flatten(params['g1'][layer_idx]), dim=0)
        gS = torch.unsqueeze(torch.flatten(params['gS'][layer_idx]), dim=0)
        k = 1/(params['epoch']+1)
        if params['epoch']==0:
            new_trace = np.trace(torch.mm((g1 - gS).t(), (g1 - gS)))
        else:
            old_trace = self.noise_covariance_traces[str(layer_idx)]
            new_trace = (1-k) * old_trace + k * np.trace(torch.mm((g1 - gS).t(),(g1 - gS)))

        self.noise_covariance_traces.update({str(layer_idx): new_trace})
        print("traces:",self.noise_covariance_traces)

        # print(torch.eig(C_new))
        # exit()
        # B = np.linalg.cholesky(C_new)
        # return B

        return self.noise_covariance_traces

    def update_learning_rate(self, layer_params, outputs, labels, optimizer_params, layer_idx):
        """
        Computes the optimal layer-wise learning rate at current time, for performing constant SGD on a minibatch of
        samples.
        """
        batch_size = len(outputs)
        n_layer_weights = np.prod([x for x in layer_params.shape])
        total_n_samples = optimizer_params['n_training_samples']
        noise_covariance_traces = self.update_traces(outputs=outputs, labels=labels, layer_params=layer_params,
                                         params=optimizer_params, layer_idx=layer_idx)

        optimal_lr = (2 * batch_size * n_layer_weights) / (total_n_samples * noise_covariance_traces[str(layer_idx)])
        return optimal_lr

    def step(self, outputs, labels, optimizer_params):
        loss = None
        weights = self.param_groups[0]

        for layer_idx, layer_params in enumerate(weights['params']):
            if layer_params.grad is None:
                continue
            gradient = layer_params.grad.data
            lr = self.update_learning_rate(layer_params=layer_params, outputs=outputs, labels=labels,
                                           optimizer_params=optimizer_params, layer_idx=layer_idx)
            layer_params.data -= -lr * gradient

        return loss