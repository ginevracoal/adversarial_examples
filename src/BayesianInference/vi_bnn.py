### TODO remove, deprecated

import sys
sys.path.append(".")
from directories import *
import pyro
from BayesianInference.bnn import BNN
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
from pyro import poutine
from utils import *
import pyro.optim as pyroopt
from pyro.optim import PyroOptim
import random
import torch
from BayesianInference.pyro_utils import data_loaders
from BayesianInference.adversarial_attacks import *
import argparse


class VI_BNN(BNN):
    def __init__(self, input_shape, device):
        self.input_size = input_shape[0]*input_shape[1]*input_shape[2]
        super(VI_BNN, self).__init__(input_size=self.input_size, device=device)

    def forward(self, inputs, n_samples):
        sampled_models = self.sample_models(n_samples)
        predictions = [model(inputs.to(self.device)).data for model in sampled_models]
        # std = torch.std(torch.stack(one_hot_predictions), 0)
        # predicted_classes = mean.argmax(-1)
        return torch.stack(predictions, dim=0)

    def infer_parameters(self, train_loader, n_samples, lr, n_epochs):
        print("\nSVI inference.")
        # optim = pyroopt.SGD({'lr': lr, 'momentum': 0.9, 'nesterov': True})
        optim = pyroopt.Adam({"lr": lr})
        elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, optim, loss=elbo, num_samples=n_samples)

        loss_list = []
        accuracy_list = []
        pyro.clear_param_store()
        for i in range(n_epochs):
            accuracy = 0
            total_loss = 0.0
            total = 0.0
            correct = 0.0

            for images, labels in train_loader:
                images = images.to(self.device).view(-1,self.input_size)
                labels = labels.to(self.device)
                loss = svi.step(inputs=images, labels=labels)
                total_loss += loss / len(train_loader.dataset)
                total += labels.size(0)
                pred = self.forward(n_samples=1, inputs=images)
                print(pred)
                correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
                accuracy = 100 * correct / total
                # print(pyro.get_param_store().get_param("fc1w_mu"))

            print(f"[Epoch {i + 1}]\t loss: {total_loss:.2f} \t accuracy: {accuracy:.2f}")
            loss_list.append(total_loss)
            accuracy_list.append(accuracy)

        print("\nlearned params =", list(pyro.get_param_store().get_all_param_names()))
        return {'loss':loss_list, 'accuracy':accuracy_list}

    def sample_models(self, n_samples):
        random.seed(0)
        sampled_models = [self.guide(None, None) for _ in range(n_samples)]
        return sampled_models

    def save(self, filename, relative_path=RESULTS):
        filepath = relative_path+"bnn/"+filename+".pr"
        os.makedirs(os.path.dirname(relative_path+"bnn/"), exist_ok=True)
        print("\nSaving params: ", filepath)
        pyro.get_param_store().save(filepath)

    def load(self, filename, relative_path=TRAINED_MODELS):
        filepath = relative_path+"bnn/"+filename+".pr"
        print("\nLoading params: ", filepath)
        pyro.get_param_store().load(filepath)

# === MAIN EXECUTIONS ===

def test_conjecture(dataset_name, n_samples, n_inputs, device):
    random.seed(0)

    # load bayesian model
    _, _, data_format, input_shape = data_loaders(dataset_name=dataset_name, batch_size=1, n_inputs=1)
    pyro.clear_param_store()
    bayesnn = VI_BNN(input_shape=input_shape, device=device)
    if dataset_name == "mnist":
        # trained_model = "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=400"
        bayesnn.load(filename=trained_model, relative_path=TRAINED_MODELS)

    # evaluate on test samples
    _, test_loader, _, _ = data_loaders(dataset_name=dataset_name, batch_size=1, n_inputs=n_inputs)
    # bayesnn.evaluate(test_loader=test_loader, n_samples=n_samples)

    # compute expected loss gradients
    exp_loss_gradients = expected_loss_gradients(model=bayesnn, n_samples=n_samples, data_loader=test_loader,
                                                 device="cuda", mode="hidden")

    filename = "hidden_vi_" + str(dataset_name) + "_inputs=" + str(n_inputs) + "_samples=" + str(n_samples)
    plot_heatmap(columns=exp_loss_gradients, path=RESULTS + "bnn/", filename=filename + "_heatmap.png",
                 xlab="pixel idx", ylab="image idx", title="Expected loss gradients on {} samples".format(n_samples))


def infer_parameters(dataset_name, n_inputs, lr, n_epochs, device, n_samples):
    random.seed(0)
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=128, n_inputs=n_inputs)
    pyro.clear_param_store()
    bayesnn = VI_BNN(input_shape=input_shape, device=device)
    filename = "hidden_vi_" + str(dataset_name) + "_inputs=" + str(n_inputs) + \
                    "_lr=" + str(lr) + "_epochs=" + str(n_epochs)
    start = time.time()
    dict = bayesnn.infer_parameters(train_loader=train_loader, n_epochs=n_epochs, lr=lr, n_samples=10)
    execution_time(start=start, end=time.time())
    plot_loss_accuracy(dict, path=RESULTS+"bnn/"+filename+".png")
    bayesnn.save(filename=filename)
    bayesnn.evaluate(test_loader=test_loader, n_samples=n_samples)


def main(args):
    infer_parameters(dataset_name=args.dataset_name, n_inputs=args.inputs, n_samples=args.samples,
                     lr=args.lr, n_epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser(description="VI Bayesian Neural Network using Pyro HiddenLayer module.")

    parser.add_argument("-n", "--inputs", nargs="?", default=100, type=int)
    parser.add_argument("--epochs", nargs='?', default=10, type=int)
    parser.add_argument("--samples", nargs='?', default=3, type=int)
    parser.add_argument("--dataset_name", nargs='?', default="mnist", type=str)
    parser.add_argument("--lr", nargs='?', default=0.002, type=float)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')

    main(args=parser.parse_args())