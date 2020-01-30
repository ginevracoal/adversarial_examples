import sys

sys.path.append(".")
from directories import *
import argparse


from utils import execution_time, plot_loss_accuracy

import pyro
import random
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as pyroopt
# from BayesianInference.plots.plot_utils import *
from BayesianInference.hidden_bnn import HiddenBNN
# from BayesianInference.adversarial_attacks import *
from BayesianInference.pyro_utils import data_loaders
# from BayesianInference.loss_gradients import *


DEBUG=False


hidden_vi_models = [
    # mnist
    {"idx":0, "filename": "hidden_vi_mnist_inputs=10000_lr=0.0002_epochs=100", "activation": "softmax",
     "dataset": "mnist", "architecture": "fully_connected"}, # pochi input, 75% test
    {"idx":1, "filename": "hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=100", "activation": "softmax",
    "dataset": "mnist", "architecture": "fully_connected"}, # overfitta, 85% test
    {"idx":2, "filename": "hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=11", "activation": "softmax",
    "dataset": "mnist", "architecture": "fully_connected"}, # 85% test
    # fashion mnist
    {"idx":3, "filename": "hidden_vi_fashion_mnist_inputs=100_lr=0.0002_epochs=800", "activation": "softmax",
    "dataset": "fashion_mnist", "architecture": "fully_connected"}, # 74% train
    {"idx":4, "filename":"hidden_vi_fashion_mnist_inputs=500_lr=0.0002_epochs=500","activation":"softmax",
    "dataset":"fashion_mnist", "architecture":"fully_connected"}, # 85% train, 75% test
    {"idx":5, "filename":"hidden_vi_fashion_mnist_inputs=1000_lr=5e-05_epochs=600","activation":"softmax",
    "dataset":"fashion_mnist", "architecture":"fully_connected"}, # 83% train, 73% test
]


class VI_BNN(HiddenBNN):
    def __init__(self, input_shape, device, architecture="fully_connected", activation="softmax"):
        self.input_size = input_shape[0]*input_shape[1]*input_shape[2]
        self.activation = activation
        self.loss = "crossentropy" if activation == "softmax" else "nllloss"
        self.hidden_size = 512
        self.n_classes = 10
        super(VI_BNN, self).__init__(input_size=self.input_size, device=device, activation=self.activation,
                                     hidden_size=self.hidden_size, architecture=architecture)

    def get_filename(self, dataset_name, n_inputs, lr, n_epochs):
        return "hidden_vi_" + str(dataset_name) + "_inputs=" + str(n_inputs) + \
                "_lr=" + str(lr) + "_epochs=" + str(n_epochs)

    def infer_parameters(self, dataset_name, train_loader, lr, n_epochs, seed=0):
        random.seed(seed)
        filename = self.get_filename(n_inputs=len(train_loader.dataset), lr=lr, n_epochs=n_epochs,
                                     dataset_name=dataset_name)
        print("\nSVI BNN:", filename)
        # optim = pyroopt.SGD({'lr': lr, 'momentum': 0.9, 'nesterov': True})
        optim = pyroopt.Adam({"lr": lr})
        elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, optim, loss=elbo)

        loss_list = []
        accuracy_list = []
        start = time.time()
        for i in range(n_epochs):
            accuracy = 0
            total_loss = 0.0
            total = 0.0
            correct = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                # svi.step() = take a gradient step on the loss function
                # images and labels are passed to model() and guide()
                loss = svi.step(inputs=images,#.view(-1,self.input_size),
                                labels=labels)
                total_loss += loss / len(train_loader.dataset)
                total += labels.size(0)
                # forward computes the average output on n_samples samples of the network
                avg_pred = self.forward(images, n_samples=3).mean(0)
                pred = avg_pred.argmax(-1).to(self.device)
                correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
                accuracy = 100 * correct / total

                if DEBUG:
                    print(images.shape)
                    print("\nimages.shape = ", images.view(-1, self.input_size).shape)
                    # np.set_printoptions(precision=2)
                    print(f"logits[0] = {avg_pred[0].cpu().detach().numpy()}")
                    print("check prob dist:", avg_pred.sum(1))
                    exit()

            print(f"[Epoch {i + 1}]\t loss: {total_loss:.2f} \t accuracy: {accuracy:.2f}")

            loss_list.append(total_loss)
            accuracy_list.append(accuracy)
        execution_time(start=start, end=time.time())

        learned_params = pyro.get_param_store().get_all_param_names()
        print(f"\nlearned params = {learned_params}")
        self.save(filename=filename)

        if DEBUG:
            if self.architecture == "fully_connected":
                print("a1_mean", pyro.get_param_store()["a1_mean"])
            elif self.architecture == "convolutional":
                print("conv1w_mu", pyro.get_param_store()["conv1w_mu"].flatten())

        plot_loss_accuracy({'loss':loss_list, 'accuracy':accuracy_list}, path=RESULTS + "bnn/" + filename + ".png")
        return self

    def load_posterior(self, posterior_name, activation="softmax", relative_path=TRAINED_MODELS):
        posterior = self.load(filename=posterior_name, relative_path=relative_path)
        posterior.activation = activation
        return posterior


def main(args):

    # === load ===

    # model, relative_path = (hidden_vi_models[0], TRAINED_MODELS)
    #
    # train_loader, test_loader, data_format, input_shape = \
    #     data_loaders(dataset_name=model["dataset"], batch_size=32, n_inputs=args.inputs, shuffle=True)
    #
    # bayesnn = VI_BNN(input_shape=input_shape, device=args.device, architecture=model["architecture"],
    #                  activation=model["activation"])
    # posterior = bayesnn.load_posterior(posterior_name=model["filename"], relative_path=RESULTS,
    #                                        activation=model["activation"])
    # posterior.evaluate(data_loader=train_loader, n_samples=args.samples)

    # === train ===

    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=128, n_inputs=args.inputs, shuffle=True)

    # train_loader = slice_data_loader(data_loader=train_loader, slice_size=100)
    # test_loader = slice_data_loader(data_loader=test_loader, slice_size=1000)

    bayesnn = VI_BNN(input_shape=input_shape, device=args.device, architecture=args.architecture,
                     activation=args.activation)
    posterior = bayesnn.infer_parameters(train_loader=train_loader, lr=args.lr, n_epochs=args.epochs,
                                         dataset_name=args.dataset)

    posterior.evaluate(data_loader=test_loader, n_samples=args.samples)



if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser(description="VI Bayesian Neural Network using Pyro HiddenLayer module.")

    parser.add_argument("-n", "--inputs", nargs="?", default=10, type=int)
    parser.add_argument("--epochs", nargs='?', default=10, type=int)
    parser.add_argument("--samples", nargs='?', default=3, type=int)
    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--architecture", nargs='?', default="fully_connected", type=str,
                        help='use "fully_connected" or "convolutional"')
    parser.add_argument("--activation", nargs='?', default="softmax", type=str)
    parser.add_argument("--lr", nargs='?', default=0.002, type=float)
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')

    main(args=parser.parse_args())