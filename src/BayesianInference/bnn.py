from directories import *
import os
from torch import nn
import torch
import pyro
from pyro.distributions import OneHotCategorical, Normal


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        output = self.fc1(x)
        output = torch.tanh(output)
        output = self.fc2(output)
        output = torch.tanh(output)
        output = self.fc3(output)
        output = torch.tanh(output)
        output = self.out(output)
        output = torch.sigmoid(output)
        return output


class BNN(nn.Module):
    def __init__(self, input_size, device):
        super(BNN, self).__init__()
        self.hidden_size = 1024
        self.n_classes = 10
        self.net = NN(input_size=input_size, hidden_size=self.hidden_size, n_classes=self.n_classes)
        self.device = device
        self.input_size = input_size

    def model(self, inputs, labels=None):
        flat_inputs = inputs.view(-1, self.input_size)
        net = self.net

        fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
        fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))

        fc2w_prior = Normal(loc=torch.zeros_like(net.fc2.weight), scale=torch.ones_like(net.fc2.weight))
        fc2b_prior = Normal(loc=torch.zeros_like(net.fc2.bias), scale=torch.ones_like(net.fc2.bias))

        fc3w_prior = Normal(loc=torch.zeros_like(net.fc3.weight), scale=torch.ones_like(net.fc3.weight))
        fc3b_prior = Normal(loc=torch.zeros_like(net.fc3.bias), scale=torch.ones_like(net.fc3.bias))

        outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight))
        outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))

        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
                  'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior,
                  'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior,
                  'out.weight': outw_prior, 'out.bias': outb_prior}

        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", net, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()
        # run the regressor forward conditioned on data
        log_softmax = nn.Softmax(dim=1)
        logits = log_softmax(lifted_reg_model(flat_inputs))
        # logits = lifted_reg_model(flat_inputs)
        # condition on the observed data
        cond_model = pyro.sample("obs", OneHotCategorical(logits=logits), obs=labels)
        return cond_model

    def guide(self, inputs, labels=None):
        net = self.net
        softplus = torch.nn.Softplus()

        # First layer weights
        fc1w_mu_param = pyro.param("fc1w_mu", torch.randn_like(net.fc1.weight))
        fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", torch.randn_like(net.fc1.weight)))
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
        # First layer bias
        fc1b_mu_param = pyro.param("fc1b_mu", torch.randn_like(net.fc1.bias))
        fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", torch.randn_like(net.fc1.bias)))
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

        # Second layer weights
        fc2w_mu_param = pyro.param("fc2w_mu", torch.randn_like(net.fc2.weight))
        fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", torch.randn_like(net.fc2.weight)))
        fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
        # Second layer bias
        fc2b_mu_param = pyro.param("fc2b_mu", torch.randn_like(net.fc2.bias))
        fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", torch.randn_like(net.fc2.bias)))
        fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)

        # Third layer weights
        fc3w_mu_param = pyro.param("fc3w_mu", torch.randn_like(net.fc3.weight))
        fc3w_sigma_param = softplus(pyro.param("fc3w_sigma", torch.randn_like(net.fc3.weight)))
        fc3w_prior = Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param)
        # Third layer bias
        fc3b_mu_param = pyro.param("fc3b_mu", torch.randn_like(net.fc3.bias))
        fc3b_sigma_param = softplus(pyro.param("fc3b_sigma", torch.randn_like(net.fc3.bias)))
        fc3b_prior = Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param)

        # Output layer weights
        outw_mu_param = pyro.param("outw_mu", torch.randn_like(net.out.weight))
        outw_sigma_param = softplus(pyro.param("outw_sigma", torch.randn_like(net.out.weight)))
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent()
        # Output layer bias
        outb_mu_param = pyro.param("outb_mu", torch.randn_like(net.out.bias))
        outb_sigma_param = softplus(pyro.param("outb_sigma", torch.randn_like(net.out.bias)))
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)

        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
                  'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior,
                  'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior,
                  'out.weight': outw_prior, 'out.bias': outb_prior}

        lifted_module = pyro.random_module("module", net, priors)
        return lifted_module()

    def infer_parameters(self, train_loader, lr, n_epochs):
        raise NotImplementedError

    def forward(self, inputs, n_samples=100):
        sampled_models = [self.guide(None, None) for _ in range(len(inputs))]
        one_hot_predictions = [model(inputs).data for model in sampled_models]
        mean = torch.mean(torch.stack(one_hot_predictions), 0)
        std = torch.std(torch.stack(one_hot_predictions), 0)
        predicted_classes = mean.argmax(-1)
        return predicted_classes

    def save(self, filename, relative_path=RESULTS):
        filepath = relative_path+"bnn/"+filename+".pr"
        os.makedirs(os.path.dirname(relative_path+"bnn/"), exist_ok=True)
        print("\nSaving params: ", filepath)
        pyro.get_param_store().save(filepath)

    def load(self, filename, relative_path=TRAINED_MODELS):
        filepath = relative_path+"bnn/"+filename+".pr"
        print("\nLoading params: ", filepath)
        pyro.get_param_store().load(filepath)

    def evaluate(self, test_loader):
        total = 0.0
        correct = 0.0
        for images, labels in test_loader:
            pred = self.forward(images.to(self.device).view(-1, self.input_size))
            total += labels.size(0)
            correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
        accuracy = 100 * correct / total
        print(f"\nTest accuracy: {accuracy:.5f}")
