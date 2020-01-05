import sys
sys.path.append(".")
from directories import *
import pyro
import torch
from BayesianInference.bnn import BNN
from BayesianSGD.sgd import PyroSGD
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
from pyro import poutine
from torch.utils.data import DataLoader
from utils import *
import pyro.optim as pyroopt
from pyro.optim import PyroOptim


class VI_BNN(BNN):
    def __init__(self, dataset_name, input_shape, data_format, test):
        super(VI_BNN, self).__init__(dataset_name=dataset_name, input_shape=input_shape, data_format=data_format,
                                     test=test)

    def infer_parameters(self, train_loader, device, lr=0.01, momentum=0.9, num_epochs=30):
        # optim = PyroSGD(params=list(self.net.parameters()), lr=lr)
        sgd = torch.optim.SGD(params=list(self.net.parameters()), lr=lr)
        optim = PyroOptim(optim_constructor=sgd, optim_args={'lr': lr, 'momentum': momentum, 'nesterov': True})
        # optim = pyroopt.SGD({'lr': lr, 'momentum': momentum, 'nesterov': True})

        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())
        kl_factor = train_loader.batch_size / len(train_loader.dataset)
        for i in range(num_epochs):
            total_loss = 0.0
            total = 0.0
            correct = 0.0
            for images, labels in train_loader:
                loss = svi.step(inputs=images.to(device), labels=labels.to(device), kl_factor=kl_factor)
                print("here")
                exit()
                pred = self.forward(images.to(device), n_samples=1).mean(0)
                total_loss += loss / len(train_loader.dataset)
                total += labels.size(0)
                correct += (pred.argmax(-1) == labels.to(device)).sum().item()
                param_store = pyro.get_param_store()
            print(f"[Epoch {i + 1}] loss: {total_loss:.5E} accuracy: {correct / total * 100:.5f}")

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
            correct += (pred.argmax(-1) == labels.cuda()).sum().item()
        print(f"Test accuracy: {correct / total * 100:.5f}")


def main(device="cpu"):
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = \
        load_dataset(dataset_name="mnist", test=True, n_samples=1000)
    pyro.clear_param_store()
    bayesnn = VI_BNN(dataset_name="mnist", data_format=data_format, input_shape=input_shape, test=True)

    train_data = DataLoader(dataset=list(zip(x_train,y_train)), batch_size=128)
    bayesnn.infer_parameters(train_loader=train_data, device=device, num_epochs=30, lr=0.002)

    test_data = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=128)
    bayesnn.evaluate_test(test_loader=test_data, device=device)


if __name__ == "__main__":
    main()