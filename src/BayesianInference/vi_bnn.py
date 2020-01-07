import sys
sys.path.append(".")
from directories import *
import pyro
from BayesianInference.bnn import BNN
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
from pyro import poutine
from torch.utils.data import DataLoader
from utils import *
import pyro.optim as pyroopt
from pyro.optim import PyroOptim
import random


class VI_BNN(BNN):
    def __init__(self, dataset_name, input_shape, data_format, test):
        super(VI_BNN, self).__init__(dataset_name=dataset_name, input_shape=input_shape, data_format=data_format,
                                     test=test)

    def infer_parameters(self, train_loader, device, lr=0.01, momentum=0.9, num_epochs=30):
        optim = pyroopt.SGD({'lr': lr, 'momentum': momentum, 'nesterov': True})
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())
        kl_factor = train_loader.batch_size / len(train_loader.dataset)

        loss_list = []
        accuracy_list = []
        for i in range(num_epochs):
            accuracy = 0
            total_loss = 0.0
            total = 0.0
            correct = 0.0

            for images, labels in train_loader:
                loss = svi.step(inputs=images.to(device), labels=labels.to(device), kl_factor=kl_factor)
                pred = self.forward(images.to(device), n_samples=1).mean(0)
                total_loss += loss / len(train_loader.dataset)
                total += labels.size(0)
                correct += (pred.argmax(-1) == onehot_to_labels(labels).to(device)).sum().item()
                accuracy = correct / total * 100
                param_store = pyro.get_param_store()
            print(f"[Epoch {i + 1}] loss: {total_loss:.2f} accuracy: {accuracy:.2f}")
            loss_list.append(total_loss)
            accuracy_list.append(accuracy)
        return {'loss':loss_list, 'accuracy':accuracy_list}

    def forward(self, inputs, n_samples=10):
        res = []
        for i in range(n_samples):
            t = poutine.trace(self.guide).get_trace(inputs)
            res.append(t.nodes['logits']['value'])
        return torch.stack(res, dim=0)

    def evaluate_test(self, test_loader, device):
        total = 0.0
        correct = 0.0
        for images, labels in test_loader:
            pred = self.forward(images.to(device).view(-1, 784), n_samples=1)
            total += labels.size(0)
            correct += (pred.argmax(-1) == onehot_to_labels(labels).to(device)).sum().item()
        print(f"Test accuracy: {correct / total * 100:.5f}")


def main(dataset_name, lr, n_epochs, device, test, seed):
    random.seed(seed)
    batch_size = 128

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = \
        load_dataset(dataset_name=dataset_name, test=test)
    pyro.clear_param_store()
    bayesnn = VI_BNN(dataset_name=dataset_name, data_format=data_format, input_shape=input_shape, test=test)

    train_data = DataLoader(dataset=list(zip(x_train,y_train)), batch_size=batch_size)

    dict = bayesnn.infer_parameters(train_loader=train_data, device=device, num_epochs=n_epochs, lr=lr)
    plot_loss_accuracy(dict, path=RESULTS+"plot.png")

    test_data = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=batch_size)
    bayesnn.evaluate_test(test_loader=test_data, device=device)


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        lr = float(sys.argv[2])
        n_epochs = int(sys.argv[3])
        device = sys.argv[4]
        test = eval(sys.argv[5])
        seed = int(sys.argv[6])

    except IndexError:
        dataset_name = input("\nChoose a dataset: ")
        lr = input("\nSet the learning rate: ")
        n_epochs = input("\nSet the number of epochs: ")
        device = input("\nChoose a device (cpu/gpu): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        seed = input("\nSet a training seed (type=int): ")

    main(dataset_name, lr, n_epochs, device, test, seed)