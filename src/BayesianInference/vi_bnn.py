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
from BayesianInference.bnn import data_loaders


class VI_BNN(BNN):
    def __init__(self, dataset_name, input_shape, data_format, device):
        super(VI_BNN, self).__init__(dataset_name=dataset_name, input_shape=input_shape, data_format=data_format,
                                     device=device)

    def infer_parameters(self, train_loader, lr=0.01, momentum=0.9, num_epochs=30):
        print("\nSVI inference.")
        optim = pyroopt.SGD({'lr': lr, 'momentum': momentum, 'nesterov': True})
        svi = SVI(self.model, self.guide, optim, loss=TraceMeanField_ELBO())
        kl_factor = train_loader.batch_size / len(train_loader.dataset)

        loss_list = []
        accuracy_list = []
        for i in range(num_epochs):
            accuracy = 0
            total_loss = 0.0
            total = 0.0
            correct = 0.0

            for images, labels in train_loader:
                loss = svi.step(inputs=images.to(self.device), labels=labels.to(self.device), kl_factor=kl_factor)
                pred = self.forward(images.to(self.device), n_samples=1).mean(0)
                total_loss += loss / len(train_loader.dataset)
                total += labels.size(0)
                correct += (pred.argmax(-1) == labels.to(self.device)).sum().item()
                accuracy = correct / total * 100

            print(f"[Epoch {i + 1}]\t loss: {total_loss:.2f} \t accuracy: {accuracy:.2f}")
            loss_list.append(total_loss)
            accuracy_list.append(accuracy)

        print("\nlearned params =", list(pyro.get_param_store().get_all_param_names()))
        return {'loss':loss_list, 'accuracy':accuracy_list}

    def forward(self, inputs, n_samples=10):
        res = []
        for i in range(n_samples):
            t = poutine.trace(self.guide).get_trace(inputs)
            res.append(t.nodes['logits']['value'])
        return torch.stack(res, dim=0)


def main(dataset_name, n_samples, lr, n_epochs, device, seed):
    random.seed(seed)
    batch_size = 128
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=batch_size, n_samples=n_samples)

    pyro.clear_param_store()
    bayesnn = VI_BNN(dataset_name=dataset_name, data_format=data_format, input_shape=input_shape, device=device)
    dict = bayesnn.infer_parameters(train_loader=train_loader, num_epochs=n_epochs, lr=lr)
    bayesnn.evaluate_test(test_loader=test_loader)

    filename = "vi_"+str(dataset_name)+"_samples="+str(n_samples)+"_lr="+str(lr)+"_epochs="+str(n_epochs)+"_seed="+str(seed)
    bayesnn.save(filename=filename)
    # bayesnn.load(filename=filename)
    # bayesnn.evaluate_test(test_loader=test_data, device=device)

    plot_loss_accuracy(dict, path=RESULTS+"bnn/"+filename+".png")


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        n_samples = int(sys.argv[2])
        lr = float(sys.argv[3])
        n_epochs = int(sys.argv[4])
        device = sys.argv[5]
        seed = int(sys.argv[6])

    except IndexError:
        dataset_name = input("\nChoose a dataset: ")
        n_samples = input("\nChoose the number of samples (type=int): ")
        lr = input("\nSet the learning rate: ")
        n_epochs = input("\nSet the number of epochs: ")
        device = input("\nChoose a device (cpu/gpu): ")
        seed = input("\nSet a training seed (type=int): ")

    main(dataset_name, n_samples, lr, n_epochs, device, seed)