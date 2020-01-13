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


class VI_BNN(BNN):
    def __init__(self, input_shape, device):
        self.input_size = input_shape[0]*input_shape[1]*input_shape[2]
        super(VI_BNN, self).__init__(input_size=self.input_size, device=device)

    def infer_parameters(self, train_loader, lr=0.001, n_epochs=30):
        print("\nSVI inference.")
        # optim = pyroopt.SGD({'lr': lr, 'momentum': 0.9, 'nesterov': True})
        optim = pyroopt.Adam({"lr": lr})
        elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, optim, loss=elbo, num_samples=1000)

        loss_list = []
        accuracy_list = []
        pyro.clear_param_store()
        for i in range(n_epochs):
            accuracy = 0
            total_loss = 0.0
            total = 0.0
            correct = 0.0

            for images, labels in train_loader:
                loss = svi.step(inputs=images.to(self.device).view(-1,self.input_size),
                                labels=labels.to(self.device))
                total_loss += loss / len(train_loader.dataset)
                total += labels.size(0)
                sampled_model = self.guide(None, None)
                pred = self.predict(sampled_models=[sampled_model],
                                    inputs=images.to(self.device).view(-1,self.input_size))
                correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
                accuracy = 100 * correct / total
                # print(pyro.get_param_store().get_param("fc1w_mu"))

            print(f"[Epoch {i + 1}]\t loss: {total_loss:.2f} \t accuracy: {accuracy:.2f}")
            loss_list.append(total_loss)
            accuracy_list.append(accuracy)

        print("\nlearned params =", list(pyro.get_param_store().get_all_param_names()))
        return {'loss':loss_list, 'accuracy':accuracy_list}

    def sample_models(self, n_samples):
        sampled_models = [self.guide(None, None) for _ in range(n_samples)]
        return sampled_models

def main(dataset_name, n_samples, lr, n_epochs, device, seed=0):
    random.seed(seed)
    batch_size = 128
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=batch_size, n_inputs=n_samples)
    filename = "vi_" + str(dataset_name) + "_samples=" + str(n_samples) + "_lr=" + str(lr) + "_epochs=" + str(
               n_epochs)

    ## === infer params ===
    pyro.clear_param_store()
    bayesnn = VI_BNN(input_shape=input_shape, device=device)

    # dict = bayesnn.infer_parameters(train_loader=train_loader, n_epochs=n_epochs, lr=lr)
    # plot_loss_accuracy(dict, path=RESULTS+"bnn/"+filename+".png")
    # bayesnn.save(filename=filename)

    bayesnn.load(filename=filename, relative_path=RESULTS)

    ## === evaluate ===
    sampled_models = bayesnn.sample_models(n_samples)
    bayesnn.evaluate(test_loader=test_loader, sampled_models=sampled_models)

    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=1, n_inputs=n_samples)

    # attack_nn(model=bayesnn.guide(None, None), data_loader=test_loader)

    # attack_bnn(model=bayesnn, n_samples=3, data_loader=test_loader)

    expected_loss_gradients(model=bayesnn, n_samples=2, data_loader=test_loader, device=device)


# todo use parser
if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        n_samples = int(sys.argv[2])
        lr = float(sys.argv[3])
        n_epochs = int(sys.argv[4])
        device = sys.argv[5]

    except IndexError:
        dataset_name = input("\nChoose a dataset: ")
        n_samples = input("\nChoose the number of samples (type=int): ")
        lr = input("\nSet the learning rate: ")
        n_epochs = input("\nSet the number of epochs: ")
        device = input("\nChoose a device (cpu/gpu): ")

    main(dataset_name, n_samples, lr, n_epochs, device)