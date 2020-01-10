from directories import *
import pyro
from pyro import poutine
import torch
from torch import nn
import pyro.contrib.bnn as bnn
import torch.nn.functional as nnf
import pyro.distributions as dist
from torch.distributions import constraints
import os


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes=10):
        super(NN, self).__init__()
        self.model = nn.Sequential(nn.Dropout(p=0.2),
                                nn.Linear(input_size, hidden_size),
                                nn.Dropout(p=0.5),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.Dropout(p=0.5),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.Dropout(p=0.5),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_size, n_classes),
                                nn.LogSoftmax(dim=-1))

    def forward(self, inputs):
        return self.model(inputs)


class BNN(nn.Module):
    def __init__(self, input_size, device):
        super(BNN, self).__init__()
        self.device = device
        self.hidden_size = 512
        self.input_size = input_size
        self.net = NN(input_size=input_size, hidden_size=self.hidden_size, n_classes=self.n_classes)

    def model(self, inputs, labels=None, kl_factor=1.0):
        size = inputs.size(0)
        flat_inputs = inputs.view(-1, self.input_size)
        # Set-up parameters for the distribution of weights for each layer `a<n>`
        a1_mean = torch.zeros(self.input_size, self.hidden_size)
        a1_scale = torch.ones(self.input_size, self.hidden_size)
        a1_dropout = torch.tensor(0.25)
        a2_mean = torch.zeros(self.hidden_size+1, self.n_classes)
        a2_scale = torch.ones(self.hidden_size+1, self.hidden_size)
        a2_dropout = torch.tensor(1.0)
        a3_mean = torch.zeros(self.hidden_size+1, self.n_classes)
        a3_scale = torch.ones(self.hidden_size+1, self.hidden_size)
        a3_dropout = torch.tensor(1.0)
        a4_mean = torch.zeros(self.hidden_size+1, self.n_classes)
        a4_scale = torch.ones(self.hidden_size+1, self.n_classes)
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

            # priors = {'a1_mean': a1_mean, 'a1_scale': a1_scale}
            # lifted_module = pyro.poutine.lift("module", self.net, priors)
            # # sample a regressor (which also samples w and b)
            # lifted_reg_model = lifted_module()
            # # run the regressor forward conditioned on data
            # log_softmax = nn.Softmax(dim=1)
            # logits = log_softmax(lifted_reg_model(flat_inputs))
            # # logits = lifted_reg_model(flat_inputs)
            # # condition on the observed data
            cond_model = pyro.sample("obs", dist.OneHotCategorical(logits=logits), obs=labels)
            return cond_model

    def guide(self, inputs, labels=None, kl_factor=1.0):
        size = inputs.size(0)
        flat_inputs = inputs.view(-1, self.input_size)
        a1_mean = pyro.param('a1_mean', 0.01 * torch.randn(self.input_size, self.hidden_size)).to(self.device)
        a1_scale = pyro.param('a1_scale', 0.1 * torch.ones(self.input_size, self.hidden_size),
                              constraint=constraints.greater_than(0.01)).to(self.device)
        a1_dropout = pyro.param('a1_dropout', torch.tensor(0.25),
                                constraint=constraints.interval(0.1, 1.0)).to(self.device)
        a2_mean = pyro.param('a2_mean', 0.01 * torch.randn(self.hidden_size+1, self.hidden_size)).to(self.device)
        a2_scale = pyro.param('a2_scale', 0.1 * torch.ones(self.hidden_size+1, self.hidden_size),
                              constraint=constraints.greater_than(0.01)).to(self.device)
        a2_dropout = pyro.param('a2_dropout', torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0)).to(self.device)
        a3_mean = pyro.param('a3_mean', 0.01 * torch.randn(self.hidden_size+1, self.hidden_size)).to(self.device)
        a3_scale = pyro.param('a3_scale', 0.1 * torch.ones(self.hidden_size+1, self.hidden_size),
                              constraint=constraints.greater_than(0.01)).to(self.device)
        a3_dropout = pyro.param('a3_dropout', torch.tensor(1.0),
                                constraint=constraints.interval(0.1, 1.0)).to(self.device)
        a4_mean = pyro.param('a4_mean', 0.01 * torch.randn(self.hidden_size+1, self.n_classes)).to(self.device)
        a4_scale = pyro.param('a4_scale', 0.1 * torch.ones(self.hidden_size+1, self.n_classes),
                               constraint=constraints.greater_than(0.01)).to(self.device)

        with pyro.plate('data', size=size):
            h1 = pyro.sample('h1',
                             bnn.HiddenLayer(flat_inputs, a1_mean, a1_dropout * a1_scale, non_linearity=nnf.leaky_relu,
                                             KL_factor=kl_factor))
            h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_dropout * a2_scale, non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_dropout * a3_scale, non_linearity=nnf.leaky_relu,
                                                   KL_factor=kl_factor))
            logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                           non_linearity=lambda x: nnf.log_softmax(x, dim=-1),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))
        # print("guide",pyro.get_param_store().get_param("a1_mean"))
        # priors = {'a1_mean': a1_mean, 'a1_scale': a1_scale}
        # lifted_module = pyro.poutine.lift(fn=self.model, prior=priors)
        # return lifted_module()

    def forward(self, inputs, n_samples):
        res = []
        for _ in range(n_samples):
            guide_trace = poutine.trace(self.guide).get_trace(inputs)
            res.append(guide_trace.nodes['logits']['value'])
        return torch.stack(res, dim=0)

    def evaluate(self, test_loader):
        total = 0.0
        correct = 0.0
        for images, labels in test_loader:
            total += labels.size(0)
            pred = self.forward(images.to(self.device), n_samples=1).mean(0).argmax(-1)
            correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
        accuracy = correct / total * 100
        print(f"\nTest accuracy: {accuracy:.5f}")

    def save(self, filename, relative_path=RESULTS):
        filepath = relative_path+"bnn/"+filename+".pr"
        os.makedirs(os.path.dirname(relative_path+"bnn/"), exist_ok=True)
        print("\nSaving params: ", filepath)
        pyro.get_param_store().save(filepath)

    def load(self, filename, relative_path=TRAINED_MODELS):
        filepath = relative_path+"bnn/"+filename+".pr"
        print("\nLoading params: ", filepath)
        pyro.get_param_store().load(filepath)