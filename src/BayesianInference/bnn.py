from directories import *
import os
from torch import nn
import torch
import pyro
from pyro.distributions import OneHotCategorical, Normal, Categorical
from pyro.nn import PyroModule


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.drop1 = nn.Dropout(p=0.2)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.drop2 = nn.Dropout(p=0.2)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        output = self.fc1(x)
        # output = self.drop1(output)
        # output = torch.relu(output)
        # output = self.fc2(output)
        # output = self.drop2(output)
        # output = torch.relu(output)
        # output = self.fc3(output)
        output = torch.relu(output)
        output = self.out(output)
        output = torch.sigmoid(output)
        return output


class BNN(nn.Module):
    def __init__(self, input_size, device):
        super(BNN, self).__init__()
        self.hidden_size = 512
        self.n_classes = 10
        self.net = NN(input_size=input_size, hidden_size=self.hidden_size, n_classes=self.n_classes)
        self.device = device
        self.input_size = input_size

    def model(self, inputs, labels=None):
        batch_size = inputs.size(0)
        flat_inputs = inputs.to(self.device).view(-1, self.input_size)
        net = self.net

        fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight)).independent(2)
        fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias)).independent(1)
        # print("fc1w_prior weights [:10] = ", net.fc1.weight)

        # fc2w_prior = Normal(loc=torch.zeros_like(net.fc2.weight), scale=torch.ones_like(net.fc2.weight)).independent(2)
        # fc2b_prior = Normal(loc=torch.zeros_like(net.fc2.bias), scale=torch.ones_like(net.fc2.bias)).independent(1)

        # fc3w_prior = Normal(loc=torch.zeros_like(net.fc3.weight), scale=torch.ones_like(net.fc3.weight)).independent(2)
        # fc3b_prior = Normal(loc=torch.zeros_like(net.fc3.bias), scale=torch.ones_like(net.fc3.bias)).independent(1)

        outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight)).independent(2)
        outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias)).independent(1)

        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
                  # 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior,
                  # 'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior,
                  'out.weight': outw_prior, 'out.bias': outb_prior}

        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", net, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()
        with pyro.plate("data", batch_size):
            # run the regressor forward conditioned on data
            log_softmax = nn.Softmax(dim=1)
            logits = log_softmax(lifted_reg_model(flat_inputs))
            # condition on the observed data
            # print(logits)
            cond_model = pyro.sample("obs", OneHotCategorical(logits=logits), obs=labels)
            return logits

    def guide(self, inputs, labels=None):
        net = self.net
        softplus = torch.nn.Softplus()
        # flat_inputs = inputs.to(self.device).view(-1, self.input_size)

        # First layer weights
        fc1w_mu_param = pyro.param("fc1w_mu", torch.randn_like(net.fc1.weight))
        fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", torch.randn_like(net.fc1.weight)))
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
        # print("fc1w_sigma_param[:10]", fc1w_sigma_param[:10])
        # First layer bias
        fc1b_mu_param = pyro.param("fc1b_mu", torch.randn_like(net.fc1.bias))
        fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", torch.randn_like(net.fc1.bias)))
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

        # # Second layer weights
        # fc2w_mu_param = pyro.param("fc2w_mu", torch.randn_like(net.fc2.weight))
        # fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", torch.randn_like(net.fc2.weight)))
        # fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
        # # Second layer bias
        # fc2b_mu_param = pyro.param("fc2b_mu", torch.randn_like(net.fc2.bias))
        # fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", torch.randn_like(net.fc2.bias)))
        # fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)
        #
        # # Third layer weights
        # fc3w_mu_param = pyro.param("fc3w_mu", torch.randn_like(net.fc3.weight))
        # fc3w_sigma_param = softplus(pyro.param("fc3w_sigma", torch.randn_like(net.fc3.weight)))
        # fc3w_prior = Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param)
        # # Third layer bias
        # fc3b_mu_param = pyro.param("fc3b_mu", torch.randn_like(net.fc3.bias))
        # fc3b_sigma_param = softplus(pyro.param("fc3b_sigma", torch.randn_like(net.fc3.bias)))
        # fc3b_prior = Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param)

        # Output layer weights
        outw_mu_param = pyro.param("outw_mu", torch.randn_like(net.out.weight))
        outw_sigma_param = softplus(pyro.param("outw_sigma", torch.randn_like(net.out.weight)))
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent()
        # Output layer bias
        outb_mu_param = pyro.param("outb_mu", torch.randn_like(net.out.bias))
        outb_sigma_param = softplus(pyro.param("outb_sigma", torch.randn_like(net.out.bias)))
        outb_prior = pyro.sample("logits", Normal(loc=outb_mu_param, scale=outb_sigma_param))

        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
                  # 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior,
                  # 'fc3.weight': fc3w_prior, 'fc3.bias': fc3b_prior,
                  'out.weight': outw_prior, 'out.bias': outb_prior}

        lifted_module = pyro.random_module("module", net, priors)
        return lifted_module()


    # def evaluate(self, sampled_models, test_loader):
    #     total = 0.0
    #     correct = 0.0
    #     for images, labels in test_loader:
    #         pred = self.predict(sampled_models=sampled_models, inputs=images.to(self.device).view(-1, self.input_size))
    #         total += labels.size(0)
    #         correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
    #     accuracy = 100 * correct / total
    #     print(f"\nTest accuracy: {accuracy:.5f}")
