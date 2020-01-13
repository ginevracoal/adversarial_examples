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
                loss = svi.step(inputs=images.view(-1,self.input_size), labels=labels)
                total_loss += loss / len(train_loader.dataset)
                total += labels.size(0)
                pred = self.forward(images.to(self.device), n_samples=n_samples).mean(0).argmax(-1)
                correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
                accuracy = 100 * correct / total

            print(f"[Epoch {i + 1}]\t loss: {total_loss:.2f} \t accuracy: {accuracy:.2f}")
            loss_list.append(total_loss)
            accuracy_list.append(accuracy)

        print("\nlearned params =", list(pyro.get_param_store().get_all_param_names()))
        return {'loss':loss_list, 'accuracy':accuracy_list}

    def predict(self, inputs, n_samples):
        predictive = Predictive(self.model, guide=self.guide, num_samples=n_samples)
        svi_samples = {k: v.reshape(n_samples).detach().to(self.device).numpy()
                       for k, v in predictive(inputs).items()
                       if k != "obs"}
        return svi_samples



def main(args):
    random.seed(0)

    # === infer params ===
    # train_loader, test_loader, data_format, input_shape = \
    #     data_loaders(dataset_name=args.dataset_name, batch_size=128, n_inputs=args.n_inputs) #1
    # pyro.clear_param_store()
    # bayesnn = VI_BNN(input_shape=input_shape, device=args.device)
    # filename = "hidden_vi_" + str(args.dataset_name) + "_inputs=" + str(args.n_inputs) + \
    #                 "_lr=" + str(args.lr) + "_epochs=" + str(args.n_epochs)
    # start = time.time()
    # dict = bayesnn.infer_parameters(train_loader=train_loader, n_epochs=args.n_epochs, lr=args.lr,
    #                                 n_samples=args.n_samples)
    # execution_time(start=start, end=time.time())
    # plot_loss_accuracy(dict, path=RESULTS+"bnn/"+filename+".png")
    # bayesnn.save(filename=filename)


    # === test conjecture ===
    # train_loader, test_loader, data_format, input_shape = \
    #     data_loaders(dataset_name=args.dataset_name, batch_size=1, n_inputs=1)
    # pyro.clear_param_store()
    # bayesnn = VI_BNN(input_shape=input_shape, device=args.device)
    # if args.dataset_name == "mnist":
    #     trained_model = "hidden_vi_mnist_inputs=60000_lr=0.02_epochs=400"
    #     bayesnn.load(filename=trained_model, relative_path=TRAINED_MODELS)
    #
    # _, test_loader, _, _ = data_loaders(dataset_name=args.dataset_name, batch_size=1, n_inputs=args.n_inputs)
    # bayesnn.evaluate(test_loader=test_loader, n_samples=args.n_samples)
    # exp_loss_gradients = expected_loss_gradients(model=bayesnn, n_samples=args.n_samples, data_loader=test_loader,
    #                                              device="cuda", mode="hidden")
    #
    # filename = "hidden_vi_" + str(args.dataset_name) + "_inputs=" + str(args.n_inputs) + "_samples=" + str(args.n_samples)
    # plot_heatmap(columns=exp_loss_gradients, path=RESULTS + "bnn/", filename=filename + "_heatmap.png",
    #              xlab="pixel idx", ylab="image idx", title="Expected loss gradients on {} samples".format(args.n_samples))

    plot_expectation_over_images(dataset_name=args.dataset_name, n_inputs=args.n_inputs,
                                 n_samples_list=[30, 50, 100, 500, 1000])


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser(description="VI Bayesian Neural Network using Pyro HiddenLayer module.")

    parser.add_argument("-n", "--n_inputs", nargs="?", default=100, type=int)
    parser.add_argument("--n_epochs", nargs='?', default=10, type=int)
    parser.add_argument("--n_samples", nargs='?', default=3, type=int)
    parser.add_argument("--dataset_name", nargs='?', default="mnist", type=str)
    parser.add_argument("--lr", nargs='?', default=0.002, type=float)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')

    main(args=parser.parse_args())