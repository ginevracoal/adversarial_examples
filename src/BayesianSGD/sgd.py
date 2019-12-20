"""
Torch implementation of basic SGD and approximate Bayesian Inference SGD from Mandt et al., 2017.
"""

from collections import defaultdict
import numpy as np
import torch
import cProfile
from torch.optim.optimizer import Optimizer

DEBUG = True

class SGD(torch.optim.SGD):
    """ Simplified version of torch.SGD """
    def __init__(self, params, lr):
        self.lr = lr
        self.weights = {}
        for layer_idx, layer_weights in enumerate(list(params)):
            self.weights.update({str(layer_idx):layer_weights})

        self.state = defaultdict(dict)
        self.param_groups = []
        self.defaults = dict(lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

        super(SGD, self).__init__(params=params, lr=lr)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None, custom_params=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        weights = self.param_groups[0]

        for layer_idx, layer_weights in enumerate(weights['params']):
            if layer_weights.grad is None:
                continue
            gradient = layer_weights.grad.data
            layer_weights.data -= self.lr * gradient

        return loss

    def zero_grad(self):
        for layer_weights in list(self.weights.values()):
            if layer_weights.grad is not None:
                layer_weights.grad.detach_()
                layer_weights.grad.zero_()


class BayesianSGD(SGD):

    def __init__(self, params, lr, loss_fn, start_updates):
        """
        Bayesian SGD training from Mandt et al. (2017) performs constant SGD, then starts updating the learning rate
        around a local minimum.
        :param params: torch network parameters, type=list
        :param lr: constant lr for the initial phase of the algorithm
        :param loss_fn: type=torch.nn.modules.loss
        :param start_updates: epoch index for starting the lr updates, type=int
        """
        super(BayesianSGD, self).__init__(params=params, lr=lr)
        self.start_updates = start_updates
        self.loss_fn = loss_fn
        self.lr = lr
        self.lr_updates = {}
        self.noise_covariance_traces = {}
        for layer_idx, layer_params in enumerate(self.param_groups[0]['params']):
            self.noise_covariance_traces.update({str(layer_idx): 0.0})

    def update_traces(self, custom_params, layer_idx, debug = False):
        """
        Updates the approximation of the noise covariance matrix at current epoch, then computes its Cholesky
        decomposition.
        :param custom_params: additional params that need to be updated, type=dict
        :param layer_idx: index of current layer, updates are saved layer-wise, type=int
        :param debug: activates debugging print calls, type=bool
        return
        :param noise_covariance_traces: updated layer-wise noise covariance traces, type=dict
        """

        g1 =  torch.flatten(custom_params['g1'][layer_idx])
        gS =  torch.flatten(custom_params['gS'][layer_idx])
        k = 1/(custom_params['epoch']-self.start_updates)
        old_trace = self.noise_covariance_traces[str(layer_idx)]

        new_trace = (1-k) * old_trace + k * torch.dot((g1 - gS).t(),(g1 - gS))

        if debug:
            print("\n(g1-gS)[:10]=", (g1-gS)[:10])

        self.noise_covariance_traces.update({str(layer_idx): new_trace})
        return self.noise_covariance_traces

    def update_learning_rate(self, layer_params, layer_idx, custom_params, debug=True):
        """
        Computes the optimal layer-wise learning rate at current time in order to perform constant SGD on a minibatch of
        samples that leads to Bayesian inference.
        :param layer_params: current layer weights, type=torch.nn.parameter.Parameter
        :param layer_idx: index of current layer, updates are saved layer-wise, type=int
        :param custom_params: additional params that need to be updated, type=dict
        :param debug: activates debugging print calls, type=bool
        return
        :param optimal_lr: optimal learning rate for current layer, type=int
        """
        batch_size = custom_params['batch_length']
        n_layer_weights = np.prod([x for x in layer_params.shape])
        total_n_samples = custom_params['n_training_samples']
        noise_covariance_trace = self.update_traces(custom_params=custom_params, layer_idx=layer_idx)[str(layer_idx)]

        # todo: optimal lr is too big
        # optimal_lr = (2 * batch_size * n_layer_weights ) / (total_n_samples * n_layer_weights * noise_covariance_trace)
        optimal_lr = (2 * noise_covariance_trace) / (n_layer_weights)  # this is wrong

        if debug:
            print("n_layer_weights=", n_layer_weights, "\ttrace = ", noise_covariance_trace, end= " ")
            print("\t\t optimal lr for layer", layer_idx, "= ", optimal_lr)

        return optimal_lr

    def step(self, closure=None, custom_params=None):
        """
        Performs gradient updates.
        :param closure: ensures compatibility with Optimizer base class
        :param custom_params: additional params that need to be updated, type=dict
        return
        :param loss:
        """
        loss = None
        if closure is not None:
            loss = closure()

        weights = self.param_groups[0]['params']
        for layer_idx, layer_params in enumerate(weights):
            if layer_params.grad is None:
                continue
            gradient = layer_params.grad.data

            if custom_params['epoch'] > self.start_updates:
                lr = self.update_learning_rate(layer_params=layer_params, layer_idx=layer_idx,
                                               custom_params=custom_params)
                self.lr_updates.update({str(layer_idx):lr})
            else:
                lr = self.lr

            layer_params.data -= lr * gradient
        return loss