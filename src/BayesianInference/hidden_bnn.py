import sys
sys.path.append(".")
from directories import *
import argparse

import pyro
from pyro import poutine
from pyro.distributions import OneHotCategorical, Normal
import torch
from torch import nn
import pyro.contrib.bnn as bnn
import torch.nn.functional as nnf
import pyro.distributions as dist
from torch.distributions import constraints
import os
from utils import save_to_pickle, load_from_pickle
import torch.optim as optim
import random
from utils import execution_time
from BayesianInference.pyro_utils import data_loaders, slice_data_loader

softplus = torch.nn.Softplus()

DEBUG=False


class NN(nn.Module):
    # todo bug in convolutional models
    def __init__(self, input_size, hidden_size, architecture, device, activation="leaky_relu", n_classes=10):
        super(NN, self).__init__()
        self.input_size = input_size
        self.architecture = architecture
        self.n_classes = n_classes
        self.dim = -1
        self.device = device

        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid
        elif activation == "tanh":
            self.activation = nn.Tanh

        if self.architecture == "fully_connected":
            self.model = nn.Sequential(
                nn.Linear(self.input_size, hidden_size),
                self.activation(),
                nn.Linear(hidden_size, hidden_size),
                self.activation(),
                nn.Linear(hidden_size, n_classes),
                nn.Softmax(dim=self.dim)
            ).to(self.device)
        elif self.architecture == "fully_connected_2":
            self.model = nn.Sequential(
                nn.Linear(self.input_size, hidden_size),
                self.activation(),
                nn.Linear(hidden_size, n_classes),
                nn.Softmax(dim=self.dim)
            ).to(self.device)
        elif self.architecture == "convolutional":
            self.conv1 = nn.Conv2d(1, 16, 1).to(self.device)
            self.conv2 = nn.Conv2d(16, 32, 3).to(self.device)
            self.dropout1 = nn.Dropout(0.25).to(self.device)
            self.fc1 = nn.Linear(32*13*13, hidden_size).to(self.device)
            self.dropout2 = nn.Dropout(0.5).to(self.device)
            self.out = nn.Linear(hidden_size, self.n_classes).to(self.device)
            self.model = self()

        print(self)
        print("\nTotal number of network weights =", sum(p.numel() for p in self.parameters()))

    def forward(self, inputs):
        if self.architecture in ["fully_connected","fully_connected_2"]:
            inputs = inputs.to(self.device).view(-1, self.input_size)
            return self.model(inputs.to(self.device))

        elif self.architecture == "convolutional":
            inputs = inputs.permute(0,3,1,2).to(self.device)
            output = self.activation()(self.conv1(inputs))
            output = self.activation()(self.conv2(output))
            output = nn.MaxPool2d((2,2))(output)
            output = self.dropout1(output)
            output = output.view(-1, output.size(1)*output.size(2)*output.size(3))
            output = self.activation()(self.fc1(output))
            output = self.dropout2(output)
            output = self.out(output)
            return output

    def train_classifier(self, dataset_name, epochs, lr, train_loader, device, input_size):
        criterion = nn.CrossEntropyLoss()
        net = self.model
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        net.train()
        start = time.time()
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0.0
            accuracy = 0.0
            total = 0.0

            for images, labels in train_loader:
                images = images.to(device).view(-1, self.input_size)
                labels = labels.to(device).argmax(-1)
                total += labels.size(0)

                optimizer.zero_grad()

                outputs = net(images).to(device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                predictions = outputs.argmax(dim=1)
                total_loss += loss.data.item() / len(train_loader.dataset)
                correct_predictions += (predictions == labels).sum()
                accuracy = 100 * correct_predictions / len(train_loader.dataset)

                # print("\noutputs: ", outputs)
                # print("\nlabels: ", labels)
                # print("\npredictions: ", predictions)
                # print(list(self.model.parameters())[0].clone())
                # exit()

            print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}", end="\t")

        execution_time(start=start, end=time.time())

        path = RESULTS +str(dataset_name)+"/nn/"
        filename = "nn_lr="+str(lr)+"_epochs="+str(epochs)+"_inputs="+str(len(train_loader.dataset))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path + filename + ".pt")


    def evaluate(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            correct_predictions = 0.0

            for images, labels in test_loader:

                images = images.to(device).view(-1, self.input_size)
                labels = labels.to(device).argmax(-1)
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == labels).sum()

            if DEBUG:
                print("\noutputs: ", outputs)
                print("\nlabels: ", labels)
                print("\npredictions: ", predictions)
                print(list(self.model.parameters())[0].clone())

            accuracy = 100 * correct_predictions / len(test_loader.dataset)
            print("\nAccuracy: %.2f%%" % (accuracy))
            return accuracy

class HiddenBNN(nn.Module):
    def __init__(self, input_size, device, activation, hidden_size, architecture):
        super(HiddenBNN, self).__init__()
        self.device = device
        self.architecture = architecture
        self.n_classes = 10
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.net = NN(input_size=input_size, hidden_size=self.hidden_size, n_classes=self.n_classes,
                      activation=activation, architecture=architecture, device=device)

        if activation == "leaky_relu":
            self.activation = nnf.leaky_relu
        elif activation == "sigmoid":
            self.activation = nnf.sigmoid#nn.Sigmoid #lambda x: nnf.sigmoid(x)
        elif activation == "tanh":
            self.activation = nnf.tanh#nn.Tanh #lambda x: nnf.tanh(x)

    def model(self, inputs, labels=None, kl_factor=1.0):
        # pyro.set_rng_seed(0)
        batch_size = inputs.size(0)
        flat_inputs = inputs.to(self.device).view(-1, self.input_size)
        if self.architecture == "fully_connected":
            # Set-up parameters for the distribution of weights for each layer `a<n>`
            a1_mean = torch.zeros(self.input_size, 32*3).to(self.device)
            a1_scale = torch.ones(self.input_size, self.hidden_size).to(self.device)
            a2_mean = torch.zeros(self.hidden_size+1, self.n_classes).to(self.device)
            a2_scale = torch.ones(self.hidden_size+1, self.hidden_size).to(self.device)
            a4_mean = torch.zeros(self.hidden_size+1, self.n_classes).to(self.device)
            a4_scale = torch.ones(self.hidden_size+1, self.n_classes).to(self.device)
            with pyro.plate('data', size=batch_size):
                # sample conditionally independent hidden layers
                h1 = pyro.sample('h1', bnn.HiddenLayer(flat_inputs, a1_mean, a1_scale,
                                                       non_linearity=self.activation, KL_factor=kl_factor))
                h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_scale,
                                                       non_linearity=self.activation, KL_factor=kl_factor))
                logits = pyro.sample('logits', bnn.HiddenLayer(h2, a4_mean, a4_scale,
                                                               non_linearity=lambda x: nnf.softmax(x, dim=self.net.dim),
                                                               KL_factor=kl_factor,
                                                               include_hidden_bias=False))
                pyro.sample("obs", dist.OneHotCategorical(logits=logits), obs=labels.to(self.device))
                return logits
        elif self.architecture == "fully_connected_2":
            # Set-up parameters for the distribution of weights for each layer `a<n>`
            a1_mean = torch.zeros(self.input_size, 32 * 3).to(self.device)
            a1_scale = torch.ones(self.input_size, self.hidden_size).to(self.device)
            a4_mean = torch.zeros(self.hidden_size + 1, self.n_classes).to(self.device)
            a4_scale = torch.ones(self.hidden_size + 1, self.n_classes).to(self.device)
            with pyro.plate('data', size=batch_size):
                # sample conditionally independent hidden layers
                h1 = pyro.sample('h1', bnn.HiddenLayer(flat_inputs, a1_mean, a1_scale,
                                                       non_linearity=self.activation, KL_factor=kl_factor))
                logits = pyro.sample('logits', bnn.HiddenLayer(h1, a4_mean, a4_scale,
                                                non_linearity=lambda x: nnf.softmax(x,dim=self.net.dim),
                                                               KL_factor=kl_factor,
                                                               include_hidden_bias=False))
                pyro.sample("obs", dist.OneHotCategorical(logits=logits), obs=labels.to(self.device))
                return logits
        elif self.architecture == "convolutional":
            net = self.net
            conv1w_prior = Normal(loc=torch.zeros_like(net.conv1.weight), scale=torch.ones_like(net.conv1.weight))
            conv1b_prior = Normal(loc=torch.zeros_like(net.conv1.bias), scale=torch.ones_like(net.conv1.bias))

            conv2w_prior = Normal(loc=torch.zeros_like(net.conv2.weight), scale=torch.ones_like(net.conv2.weight))
            conv2b_prior = Normal(loc=torch.zeros_like(net.conv2.bias), scale=torch.ones_like(net.conv2.bias))

            fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
            fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))

            outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight))
            outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))

            priors = {'conv1.weight': conv1w_prior, 'conv1.bias': conv1b_prior,
                      'conv2.weight': conv2w_prior, 'conv2.bias': conv2b_prior,
                      'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
                      'out.weight': outw_prior, 'out.bias': outb_prior}

            lifted_module = pyro.random_module("module", net, priors)
            lifted_reg_model = lifted_module()
            outputs = lifted_reg_model(inputs.to(self.device))
            logits = nnf.softmax(outputs, dim=self.net.dim)
            pyro.sample("obs", OneHotCategorical(logits=logits), obs=labels.to(self.device))
            return logits.to(self.device)

    def guide(self, inputs, labels=None, kl_factor=1.0):
        # handle old models
        if type(self.activation)==str:
            self.activation = nnf.leaky_relu

        batch_size = inputs.size(0)
        flat_inputs = inputs.to(self.device).view(-1, self.input_size)

        if self.architecture == "fully_connected":
            a1_mean = pyro.param('a1_mean', 0.01 * torch.randn(self.input_size, self.hidden_size)).to(self.device)
            a1_scale = pyro.param('a1_scale', 0.1 * torch.ones(self.input_size, self.hidden_size),
                                  constraint=constraints.greater_than(0.01)).to(self.device)
            a2_mean = pyro.param('a2_mean', 0.01 * torch.randn(self.hidden_size + 1, self.hidden_size)).to(self.device)
            a2_scale = pyro.param('a2_scale', 0.1 * torch.ones(self.hidden_size + 1, self.hidden_size),
                                  constraint=constraints.greater_than(0.01)).to(self.device)
            a4_mean = pyro.param('a4_mean', 0.01 * torch.randn(self.hidden_size + 1, self.n_classes)).to(self.device)
            a4_scale = pyro.param('a4_scale', 0.1 * torch.ones(self.hidden_size + 1, self.n_classes),
                                  constraint=constraints.greater_than(0.01)).to(self.device)

            with pyro.plate('data', size=batch_size):
                h1 = pyro.sample('h1', bnn.HiddenLayer(flat_inputs, a1_mean, a1_scale,
                                                       non_linearity=self.activation, KL_factor=kl_factor))
                h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_scale,
                                                       non_linearity=self.activation, KL_factor=kl_factor))
                pyro.sample('logits', bnn.HiddenLayer(h2, a4_mean, a4_scale,
                                                      non_linearity=lambda x: nnf.softmax(x, dim=self.net.dim),
                                                      KL_factor=kl_factor, include_hidden_bias=False))

        elif self.architecture == "fully_connected_2":
            a1_mean = pyro.param('a1_mean', 0.01 * torch.randn(self.input_size, self.hidden_size)).to(self.device)
            a1_scale = pyro.param('a1_scale', 0.1 * torch.ones(self.input_size, self.hidden_size),
                                  constraint=constraints.greater_than(0.01)).to(self.device)
            a4_mean = pyro.param('a4_mean', 0.01 * torch.randn(self.hidden_size + 1, self.n_classes)).to(self.device)
            a4_scale = pyro.param('a4_scale', 0.1 * torch.ones(self.hidden_size + 1, self.n_classes),
                                  constraint=constraints.greater_than(0.01)).to(self.device)

            with pyro.plate('data', size=batch_size):
                h1 = pyro.sample('h1', bnn.HiddenLayer(flat_inputs, a1_mean, a1_scale,
                                                       non_linearity=self.activation, KL_factor=kl_factor))
                pyro.sample('logits', bnn.HiddenLayer(h1, a4_mean, a4_scale,
                                                      non_linearity=lambda x: nnf.softmax(x, dim=self.net.dim),
                                                      KL_factor=kl_factor, include_hidden_bias=False))
        elif self.architecture == "convolutional":
            net = self.net
            # convolution layer weights
            conv1w_mu = torch.randn_like(net.conv1.weight).to(self.device)
            conv1w_sigma = torch.randn_like(net.conv1.weight).to(self.device)
            conv1w_prior = Normal(loc=pyro.param("conv1w_mu", conv1w_mu),
                                  scale=softplus(pyro.param("conv1w_sigma", conv1w_sigma)))
            # First layer bias distribution priors
            conv1b_mu = torch.randn_like(net.conv1.bias).to(self.device)
            conv1b_sigma = torch.randn_like(net.conv1.bias).to(self.device)
            conv1b_prior = Normal(loc=pyro.param("conv1b_mu", conv1b_mu),
                                  scale=softplus(pyro.param("conv1b_sigma", conv1b_sigma)))

            # convolution layer weights
            conv2w_mu = torch.randn_like(net.conv2.weight).to(self.device)
            conv2w_sigma = torch.randn_like(net.conv2.weight).to(self.device)
            conv2w_prior = Normal(loc=pyro.param("conv2w_mu", conv2w_mu),
                                  scale=softplus(pyro.param("conv2w_sigma", conv2w_sigma)))
            # First layer bias distribution priors
            conv2b_mu = torch.randn_like(net.conv2.bias).to(self.device)
            conv2b_sigma = torch.randn_like(net.conv2.bias).to(self.device)
            conv2b_prior = Normal(loc=pyro.param("conv2b_mu", conv2b_mu),
                                  scale=softplus(pyro.param("conv2b_sigma", conv2b_sigma)))

            # First layer weight distribution priors
            fc1w_mu = torch.randn_like(net.fc1.weight).to(self.device)
            fc1w_sigma = torch.randn_like(net.fc1.weight).to(self.device)
            fc1w_prior = Normal(loc=pyro.param("fc1w_mu", fc1w_mu),
                                scale=softplus(pyro.param("fc1w_sigma", fc1w_sigma)))
            # First layer bias distribution priors
            fc1b_mu = torch.randn_like(net.fc1.bias).to(self.device)
            fc1b_sigma = torch.randn_like(net.fc1.bias).to(self.device)
            fc1b_prior = Normal(loc=pyro.param("fc1b_mu", fc1b_mu),
                                scale=softplus(pyro.param("fc1b_sigma", fc1b_sigma)))
            # Output layer weight distribution priors
            outw_mu = torch.randn_like(net.out.weight).to(self.device)
            outw_sigma = torch.randn_like(net.out.weight).to(self.device)
            outw_prior = Normal(loc=pyro.param("outw_mu", outw_mu),
                                scale=softplus(pyro.param("outw_sigma", outw_sigma))).independent(1)
            # Output layer bias distribution priors
            outb_mu = torch.randn_like(net.out.bias).to(self.device)
            outb_sigma = torch.randn_like(net.out.bias).to(self.device)
            outb_prior = Normal(loc=pyro.param("outb_mu", outb_mu),
                                scale=softplus(pyro.param("outb_sigma", outb_sigma)))

            priors = {'conv1.weight': conv1w_prior, 'conv1.bias': conv1b_prior,
                      'conv2.weight': conv2w_prior, 'conv2.bias': conv2b_prior,
                      'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
                      'out.weight': outw_prior, 'out.bias': outb_prior}
            lifted_module = pyro.random_module("module", net, priors)
            lifted_reg_model = lifted_module()

            logits = nnf.softmax(lifted_reg_model(inputs.to(self.device)), dim=self.net.dim)
            return logits.to(self.device)


    def forward(self, inputs, n_samples):
        pyro.set_rng_seed(0)

        res = []

        if DEBUG:
            if self.architecture in ["fully_connected","fully_connected_2"]:
                print("a1_mean", pyro.get_param_store()["a1_mean"])
            elif self.architecture == "convolutional":
                print("conv1w_mu",pyro.get_param_store()["conv1w_mu"].flatten())

        for _ in range(n_samples):
            guide_trace = poutine.trace(self.guide).get_trace(inputs)
            if self.architecture in ["fully_connected","fully_connected_2"]:
                res.append(guide_trace.nodes['logits']['value'])
            elif self.architecture == "convolutional":
                res.append(guide_trace.nodes['_RETURN']['value'])
        res = torch.stack(res, dim=0)
        return res

    def evaluate(self, data_loader, n_samples, device="cpu"):
        total = 0.0
        correct = 0.0
        outputs = []
        for images, labels in data_loader:
            total += labels.size(0)
            output = self.forward(images.to(self.device), n_samples=n_samples)
            outputs.append(output)
            pred = output.mean(0).argmax(-1)
            labels = labels.to(self.device).argmax(-1)

            if DEBUG:
                print("\npred[:5]=", pred[:5], "\tlabels[:5]=", labels[:5])

            correct += (pred == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"accuracy on {n_samples} samples = {accuracy:.2f}")

        return {"accuracy": accuracy, "outputs":outputs}

    def save(self, filename, dataset_name, relative_path=RESULTS):
        path = relative_path+str(dataset_name)+"/bnn/"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("\nSaving params: ", path+filename+".pr")
        pyro.get_param_store().save(path+filename+".pr")

    def load(self, filename, dataset_name, relative_path=TRAINED_MODELS):
        filepath = relative_path+str(dataset_name)+"/bnn/"+filename+".pr"
        print("\nLoading params: ", filepath)
        pyro.get_param_store().load(filepath)
        return self


def main(args):
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=64, n_inputs=args.inputs, shuffle=True)

    epsilon_list = [0.1, 0.3, 0.6]

    for epsilon in epsilon_list:
        print("\n\nepsilon =", epsilon)

        for train_slice, test_slice in zip([10, 20, 50, 100]):

            train = slice_data_loader(train_loader, slice_size=train_slice)
            test = slice_data_loader(test_loader, slice_size=test_slice)

            for lr in [0.2, 0.02, 0.002]:
                for epochs in [10, 20, 50, 100]:
                    # === initialize class ===
                    input_size = input_shape[0]*input_shape[1]*input_shape[2]
                    net = NN(input_size=input_size, hidden_size=512, activation="leaky_relu",
                             device=args.device, architecture="fully_connected")
                    # path = RESULTS +str(args.dataset)+"/nn/"
                    # filename = "nn_lr="+str(lr)+"_epochs="+str(epochs)+"_inputs="+str(args.inputs)

                    # === train ===
                    net.train_classifier(epochs=epochs, lr=lr, train_loader=train, device=args.device,
                                                 input_size=input_size, dataset_name=args.dataset)
                    # os.makedirs(os.path.dirname(path), exist_ok=True)
                    # torch.save(net.model.state_dict(), path + filename + ".pt")

                    # === load ===
                    # net.model.load_state_dict(torch.load(path + filename + ".pt"))

                    # == evaluate ===
                    # acc = net.evaluate(test_loader=test,device=args.device)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser(description="Basic Neural Network.")

    parser.add_argument("-n", "--inputs", nargs="?", default=1000, type=int)
    parser.add_argument("--epochs", nargs='?', default=10, type=int)
    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--activation", nargs='?', default="leaky_relu", type=str)
    parser.add_argument("--lr", nargs='?', default=0.002, type=float)
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')

    main(args=parser.parse_args())