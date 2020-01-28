import sys
sys.path.append(".")
from directories import *
import argparse

import pyro
from pyro import poutine
from torch.autograd import Variable
import torch
from torch import nn
import pyro.contrib.bnn as bnn
import torch.nn.functional as nnf
import pyro.distributions as dist
from torch.distributions import constraints
import os
import numpy as np
from utils import save_to_pickle, load_from_pickle
import torch.optim as optim
import random
from BayesianSGD.classifier import SGDClassifier
from utils import execution_time

from BayesianInference.adversarial_attacks import attack
# from BayesianInference.plot_utils import scatterplot_accuracy_robustness
from BayesianInference.pyro_utils import data_loaders, slice_data_loader

softplus = torch.nn.Softplus()

DEBUG=False


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, dataset_name, device, activation="softmax", n_classes=10):
        super(NN, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.dim = -1
        self.device = device
        if dataset_name == "mnist":
            self.model = nn.Sequential(
                                nn.Linear(self.input_size, hidden_size),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_size, n_classes),
                                nn.Softmax(dim=self.dim) if activation == "softmax" else nn.LogSoftmax(dim=self.dim)).to(self.device)
        else:
            self.model = nn.Sequential(#nn.Dropout(p=0.2),
                                nn.Linear(self.input_size, hidden_size),
                                # nn.Dropout(p=0.5),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                # nn.Dropout(p=0.5),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                # nn.Dropout(p=0.5),
                                nn.LeakyReLU(),
                                nn.Linear(hidden_size, n_classes),
                                nn.Softmax(dim=self.dim) if activation == "softmax" else nn.LogSoftmax(dim=self.dim)).to(self.device)

    def forward(self, inputs):
        # print(self.model(inputs))
        # inputs = inputs.to(self.device).view(-1, self.input_size)
        return self.model(inputs.to(self.device))

    def train_classifier(self, epochs, lr, train_loader, device, input_size):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        self.model.train()
        start = time.time()
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0.0
            accuracy = 0.0
            total = 0.0

            for images, labels in train_loader:
                images = images.to(device).view(-1, self.input_size)
                labels = labels.to(device).argmax(-1)  # .long()
                total += labels.size(0)

                optimizer.zero_grad()

                outputs = self.model(images).to(device)
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

            print(f"[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}")

        execution_time(start=start, end=time.time())
        # return model

    def evaluate(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            correct_predictions = 0.0
            accuracy = 0.0

            for images, labels in test_loader:

                images = images.to(device).view(-1, self.input_size)
                labels = labels.to(device).argmax(-1)
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == labels).sum()

            # print("\noutputs: ", outputs)
            # print("\nlabels: ", labels)
            # print("\npredictions: ", predictions)
            # print(list(self.model.parameters())[0].clone())

            accuracy = 100 * correct_predictions / len(test_loader.dataset)
            print("\nAccuracy: %.2f%%" % (accuracy))
            return accuracy

class HiddenBNN(nn.Module):
    def __init__(self, input_size, device, activation, hidden_size, dataset_name):
        super(HiddenBNN, self).__init__()
        self.device = device
        self.dataset_name = dataset_name
        self.n_classes = 10
        self.hidden_size = hidden_size
        if activation == "softmax":
            self.activation_func = nnf.softmax
        elif activation == "log_softmax":
            self.activation_func = nnf.log_softmax
        self.input_size = input_size
        self.net = NN(input_size=input_size, hidden_size=self.hidden_size, n_classes=self.n_classes,
                      activation=activation, dataset_name=dataset_name, device=device)
        print(self.net)

    def model(self, inputs, labels=None, kl_factor=1.0):
        batch_size = inputs.size(0)
        flat_inputs = inputs.to(self.device).view(-1, self.input_size)
        if self.dataset_name == "mnist":
            # Set-up parameters for the distribution of weights for each layer `a<n>`
            a1_mean = torch.zeros(self.input_size, self.hidden_size).to(self.device)
            a1_scale = torch.ones(self.input_size, self.hidden_size).to(self.device)
            a2_mean = torch.zeros(self.hidden_size+1, self.n_classes).to(self.device)
            a2_scale = torch.ones(self.hidden_size+1, self.hidden_size).to(self.device)
            a4_mean = torch.zeros(self.hidden_size+1, self.n_classes).to(self.device)
            a4_scale = torch.ones(self.hidden_size+1, self.n_classes).to(self.device)
            with pyro.plate('data', size=batch_size):
                # sample conditionally independent hidden layers
                h1 = pyro.sample('h1', bnn.HiddenLayer(flat_inputs, a1_mean, a1_scale,
                                                       non_linearity=nnf.leaky_relu, KL_factor=kl_factor))
                h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_scale,
                                                       non_linearity=nnf.leaky_relu, KL_factor=kl_factor))
                logits = pyro.sample('logits', bnn.HiddenLayer(h2, a4_mean, a4_scale,
                                                               non_linearity=lambda x: self.activation_func(x, dim=self.net.dim),
                                                               KL_factor=kl_factor,
                                                               include_hidden_bias=False))
                pyro.sample("obs", dist.OneHotCategorical(logits=logits), obs=labels.to(self.device))
                return logits
        else:
            # Set-up parameters for the distribution of weights for each layer `a<n>`
            a1_mean = torch.zeros(self.input_size, self.hidden_size).to(self.device)
            a1_scale = torch.ones(self.input_size, self.hidden_size).to(self.device)
            # a1_dropout = torch.tensor(0.25)
            a2_mean = torch.zeros(self.hidden_size + 1, self.hidden_size).to(self.device)
            a2_scale = torch.ones(self.hidden_size + 1, self.hidden_size).to(self.device)
            # a2_dropout = torch.tensor(1.0)
            a3_mean = torch.zeros(self.hidden_size+1, self.hidden_size).to(self.device)
            a3_scale = torch.ones(self.hidden_size+1, self.hidden_size).to(self.device)
            # a3_dropout = torch.tensor(1.0)
            a4_mean = torch.zeros(self.hidden_size + 1, self.hidden_size).to(self.device)
            a4_scale = torch.ones(self.hidden_size + 1, self.hidden_size).to(self.device)
            # a5_mean = torch.zeros(self.hidden_size + 1, self.n_classes).to(self.device)
            # a5_scale = torch.ones(self.hidden_size + 1, self.hidden_size).to(self.device)
            with pyro.plate('data', size=batch_size):
                # sample conditionally independent hidden layers
                h1 = pyro.sample('h1', bnn.HiddenLayer(flat_inputs, a1_mean, a1_scale,  # a1_dropout*a1_scale,
                                                       non_linearity=nnf.leaky_relu, KL_factor=kl_factor))
                # print("h1", h1[:5], end="\t")
                h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_scale,  # a2_dropout*a2_scale
                                                       non_linearity=nnf.leaky_relu, KL_factor=kl_factor))
                h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_scale, #a3_dropout*a3_scale
                                                       non_linearity=nnf.leaky_relu, KL_factor=kl_factor))
                # h4 = pyro.sample('h4', bnn.HiddenLayer(h3, a4_mean, a4_scale, #a3_dropout*a3_scale
                #                                        non_linearity=nnf.leaky_relu, KL_factor=kl_factor))
                # logits = pyro.sample('logits', bnn.HiddenLayer(h4, a5_mean, a5_scale,
                logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                non_linearity=lambda x: self.activation_func(x, dim=self.net.dim),
                                                               KL_factor=kl_factor,
                                                               include_hidden_bias=False))

                pyro.sample("obs", dist.OneHotCategorical(logits=logits), obs=labels.to(self.device))
                return logits

    def guide(self, inputs, labels=None, kl_factor=1.0):
        batch_size = inputs.size(0)
        flat_inputs = inputs.to(self.device).view(-1, self.input_size)

        if self.dataset_name == "mnist":
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
                                                       non_linearity=nnf.leaky_relu, KL_factor=kl_factor))
                h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_scale,
                                                       non_linearity=nnf.leaky_relu, KL_factor=kl_factor))
                pyro.sample('logits', bnn.HiddenLayer(h2, a4_mean, a4_scale,
                                                      non_linearity=lambda x: self.activation_func(x, dim=self.net.dim),
                                                      KL_factor=kl_factor, include_hidden_bias=False))

        else:
            a1_mean = pyro.param('a1_mean', 0.01 * torch.randn(self.input_size, self.hidden_size)).to(self.device)
            a1_scale = pyro.param('a1_scale', 0.1 * torch.ones(self.input_size, self.hidden_size),
                                  constraint=constraints.greater_than(0.01)).to(self.device)
            # a1_dropout = pyro.param('a1_dropout', torch.tensor(0.25),
            #                         constraint=constraints.interval(0.1, 1.0)).to(self.device)
            a2_mean = pyro.param('a2_mean', 0.01 * torch.randn(self.hidden_size+1, self.hidden_size)).to(self.device)
            a2_scale = pyro.param('a2_scale', 0.1 * torch.ones(self.hidden_size+1, self.hidden_size),
                                  constraint=constraints.greater_than(0.01)).to(self.device)
            # a2_dropout = pyro.param('a2_dropout', torch.tensor(1.0),
            #                         constraint=constraints.interval(0.1, 1.0)).to(self.device)
            a3_mean = pyro.param('a3_mean', 0.01 * torch.randn(self.hidden_size+1, self.hidden_size)).to(self.device)
            a3_scale = pyro.param('a3_scale', 0.1 * torch.ones(self.hidden_size+1, self.hidden_size),
                                  constraint=constraints.greater_than(0.01)).to(self.device)
            # a3_dropout = pyro.param('a3_dropout', torch.tensor(1.0),
            #                         constraint=constraints.interval(0.1, 1.0)).to(self.device)
            a4_mean = pyro.param('a4_mean', 0.01 * torch.randn(self.hidden_size+1, self.hidden_size)).to(self.device)
            a4_scale = pyro.param('a4_scale', 0.1 * torch.ones(self.hidden_size+1, self.hidden_size),
                                   constraint=constraints.greater_than(0.01)).to(self.device)
            # a5_mean = pyro.param('a5_mean', 0.01 * torch.randn(self.hidden_size + 1, self.n_classes)).to(self.device)
            # a5_scale = pyro.param('a5_scale', 0.1 * torch.ones(self.hidden_size + 1, self.n_classes),
            #                       constraint=constraints.greater_than(0.01)).to(self.device)

            with pyro.plate('data', size=batch_size):
                h1 = pyro.sample('h1', bnn.HiddenLayer(flat_inputs, a1_mean, a1_scale,#a1_dropout * a1_scale,
                                                       non_linearity=nnf.leaky_relu, KL_factor=kl_factor))
                h2 = pyro.sample('h2', bnn.HiddenLayer(h1, a2_mean, a2_scale,#a2_dropout * a2_scale
                                                       non_linearity=nnf.leaky_relu, KL_factor=kl_factor))
                h3 = pyro.sample('h3', bnn.HiddenLayer(h2, a3_mean, a3_scale,#a3_dropout * a3_scale,
                                                       non_linearity=nnf.leaky_relu, KL_factor=kl_factor))
                # h4 = pyro.sample('h4', bnn.HiddenLayer(h3, a4_mean, a4_scale,#a3_dropout * a3_scale,
                #                                        non_linearity=nnf.leaky_relu, KL_factor=kl_factor))
                # logits = pyro.sample('logits', bnn.HiddenLayer(h4, a5_mean, a5_scale,
                logits = pyro.sample('logits', bnn.HiddenLayer(h3, a4_mean, a4_scale,
                                                               non_linearity=lambda x: self.activation_func(x, dim=self.net.dim),
                                                               KL_factor=kl_factor,
                                                               include_hidden_bias=False))

    def forward(self, inputs, n_samples):
        random.seed(0)
        res = []
        if DEBUG:
            print("a1_mean", pyro.get_param_store()["a1_mean"])
            print("a2_scale",pyro.get_param_store()["a2_scale"])
        for _ in range(n_samples):
            guide_trace = poutine.trace(self.guide).get_trace(inputs)
            res.append(guide_trace.nodes['logits']['value'])
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
        print(f"accuracy on {n_samples} samples = {accuracy:.2f}", end="\t")

        return {"accuracy": accuracy, "outputs":outputs}

    def save(self, filename, relative_path=RESULTS):
        filepath = relative_path+"bnn/"+filename+".pr"
        os.makedirs(os.path.dirname(relative_path+"bnn/"), exist_ok=True)
        print("\nSaving params: ", filepath)
        pyro.get_param_store().save(filepath)

    def load(self, filename, relative_path=TRAINED_MODELS):
        filepath = relative_path+"bnn/"+filename+".pr"
        print("\nLoading params: ", filepath)
        pyro.get_param_store().load(filepath)
        return self


def main(args):
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=64, n_inputs=args.inputs, shuffle=True)

    epsilon_list = [0.1, 0.3, 0.6]

    plot_accuracy = []
    plot_robustness = []
    plot_eps = []
    model_type = []

    for epsilon in epsilon_list:
        print("\n\nepsilon =", epsilon)

        for train_slice, test_slice in zip([10, 20, 50, 100]):

            train = slice_data_loader(train_loader, slice_size=train_slice)
            test = slice_data_loader(test_loader, slice_size=test_slice)

            for lr in [0.2, 0.02, 0.002]:
                for epochs in [10, 20, 50, 100]:
                    # === initialize class ===
                    input_size = input_shape[0]*input_shape[1]*input_shape[2]
                    net = NN(input_size=input_size, hidden_size=512, dataset_name=args.dataset, activation="softmax",
                             device=args.device)
                    # path = RESULTS +"nn/"
                    # filename = str(args.dataset)+"_nn_lr="+str(lr)+"_epochs="+str(epochs)+"_inputs="+str(args.inputs)

                    # === train ===
                    net.train_classifier(epochs=epochs, lr=lr, train_loader=train, device=args.device,
                                                 input_size=input_size)
                    # os.makedirs(os.path.dirname(path), exist_ok=True)
                    # torch.save(net.model.state_dict(), path + filename + ".pt")

                    # === load ===
                    # net.model.load_state_dict(torch.load(path + filename + ".pt"))

                    # == evaluate ===
                    # acc = net.evaluate(test_loader=test,device=args.device)

                    # == attack ==
                    attack_dict = attack(model=net.model, data_loader=test, epsilon=epsilon, device=args.device)

                    # attack_dict = load_from_pickle(path=RESULTS + "nn/"+filename)
                    plot_robustness.append(attack_dict["softmax_robustness"])
                    plot_accuracy.append(attack_dict["original_accuracy"])
                    plot_eps.append(attack_dict["epsilon"])
                    model_type.append("nn")
                    # todo loss gradient norms plot for BNNs

    idx = 0
    filename = str(args.dataset) + "_nn_attack"+str(idx)+".pkl"
    data = {"accuracy":plot_accuracy, "softmax_robustness":plot_robustness,
            "model_type":model_type, "epsilon":plot_eps}

    save_to_pickle(relative_path=RESULTS + "nn/", filename=filename, data=data)

    scatterplot_accuracy_robustness(accuracy=plot_accuracy, robustness=plot_robustness, model_type=model_type,
                                    epsilon=plot_eps)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser(description="Basic Neural Network.")

    parser.add_argument("-n", "--inputs", nargs="?", default=1000, type=int)
    parser.add_argument("--epochs", nargs='?', default=10, type=int)
    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--activation", nargs='?', default="softmax", type=str)
    parser.add_argument("--lr", nargs='?', default=0.002, type=float)
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')

    main(args=parser.parse_args())