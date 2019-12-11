"""
Torch implementation of basic SGD and approximate Bayesian Inference SGD from Mandt et al., 2017.
"""

from collections import defaultdict
from torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    """ Simplified version of torch.SGD without Nesterov momentum. """
    def __init__(self, params, lr, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SGD, self).__init__(params, defaults)

        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                gradient = p.grad.data
                if weight_decay != 0:
                    gradient.add_(weight_decay, p.data)
                p.data.add_(-group['lr'], gradient)

        return loss


# class Bayesian_SGD(SGD):
#
#     def optimal_learning_rate(self, batch_size, weights):
#         """
#         Computes the optimal learning rate for performing constant SGD on a minibatch of samples
#         """
#         # todo: compute weight matrix
#         return (2*batch_size*self.n_weights)/(self.n_samples*torch.trace(weight_matrix))
