import sys
sys.path.append(".")
from directories import *
import pyro
from BayesianInference.hidden_bnn import HiddenBNN
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO, EmpiricalMarginal
from utils import *
import pyro.optim as pyroopt
import random
from BayesianInference.adversarial_attacks import *
from BayesianInference.pyro_utils import data_loaders
import time
from utils import execution_time
import argparse
from pyro.infer import Predictive


class VI_BNN(HiddenBNN):
    def __init__(self, input_shape, device):
        self.input_size = input_shape[0]*input_shape[1]*input_shape[2]
        self.n_classes = 10
        super(VI_BNN, self).__init__(input_size=self.input_size, device=device)

    def infer_parameters(self, train_loader, lr, n_epochs, n_samples):
        print("\nSVI inference.")
        # optim = pyroopt.SGD({'lr': lr, 'momentum': 0.9, 'nesterov': True})
        optim = pyroopt.Adam({"lr": lr})#, "betas": (0.95, 0.999)})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self.model, self.guide, optim, loss=elbo)

        loss_list = []
        accuracy_list = []
        for i in range(n_epochs):
            accuracy = 0
            total_loss = 0.0
            total = 0.0
            correct = 0.0

            for images, labels in train_loader:
                images.to(self.device)
                labels.to(self.device)
                # svi.step() = take a gradient step on the loss function
                # images and labels are passed to model() and guide()
                loss = svi.step(inputs=images.view(-1,self.input_size), labels=labels)
                total_loss += loss / len(train_loader.dataset)
                total += labels.size(0)
                # forward computes the average output on n_samples samples of the network
                pred = self.forward(images.to(self.device), n_samples=1).mean(0).argmax(-1)
                correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
                accuracy = 100 * correct / total

            print(f"[Epoch {i + 1}]\t loss: {total_loss:.2f} \t accuracy: {accuracy:.2f}")
            loss_list.append(total_loss)
            accuracy_list.append(accuracy)

        print("\nlearned params =", list(pyro.get_param_store().get_all_param_names()))
        return {'loss':loss_list, 'accuracy':accuracy_list}

    # def predict(self, inputs, n_samples):
    #     predictive = Predictive(self.model, guide=self.guide, num_samples=n_samples)
    #     svi_samples = {k: v.reshape(n_samples).detach().to(self.device).numpy()
    #                    for k, v in predictive(inputs).items()
    #                    if k != "obs"}
    #     return svi_samples

# === MAIN EXECUTIONS ===

def test_conjecture(dataset_name, n_samples, n_inputs, device):
    random.seed(0)

    # load bayesian model
    _, _, data_format, input_shape = data_loaders(dataset_name=dataset_name, batch_size=1, n_inputs=1)
    pyro.clear_param_store()
    bayesnn = VI_BNN(input_shape=input_shape, device=device)
    if dataset_name == "mnist":
        # trained_model = "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=200"
        # trained_model = "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=400"
        # trained_model = "hidden_vi_mnist_inputs=60000_lr=0.002_epochs=100"
        trained_model = "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=80"
        bayesnn.load(filename=trained_model, relative_path=TRAINED_MODELS)

    # evaluate on test samples
    _, test_loader, _, _ = data_loaders(dataset_name=dataset_name, batch_size=100, n_inputs=n_inputs)
    bayesnn.evaluate(test_loader=test_loader, n_samples=n_samples)
    exit()
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
    dict = bayesnn.infer_parameters(train_loader=train_loader, n_epochs=n_epochs, lr=lr, n_samples=1)
    execution_time(start=start, end=time.time())
    plot_loss_accuracy(dict, path=RESULTS+"bnn/"+filename+".png")
    bayesnn.save(filename=filename)
    bayesnn.evaluate(test_loader=test_loader, n_samples=n_samples)


def main(args):
    #
    infer_parameters(dataset_name=args.dataset_name, n_inputs=args.inputs, n_samples=args.samples,
                     lr=args.lr, n_epochs=args.epochs, device=args.device)

    test_conjecture(dataset_name=args.dataset_name, n_samples=args.samples,
                    n_inputs=args.inputs, device=args.device)

    # plot_expectation_over_images(dataset_name=args.dataset_name, n_inputs=args.inputs,
    #                              n_samples_list=[10, 50, 100])


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser(description="VI Bayesian Neural Network using Pyro HiddenLayer module.")

    parser.add_argument("-n", "--inputs", nargs="?", default=10, type=int)
    parser.add_argument("--epochs", nargs='?', default=10, type=int)
    parser.add_argument("--samples", nargs='?', default=3, type=int)
    parser.add_argument("--dataset_name", nargs='?', default="mnist", type=str)
    parser.add_argument("--lr", nargs='?', default=0.002, type=float)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')

    main(args=parser.parse_args())