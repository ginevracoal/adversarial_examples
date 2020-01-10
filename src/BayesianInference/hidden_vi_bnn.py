import sys
sys.path.append(".")
from directories import *
import pyro
from BayesianInference.hidden_bnn import BNN
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO, EmpiricalMarginal
from utils import *
import pyro.optim as pyroopt
import random
from BayesianInference.adversarial_attacks import *
from BayesianInference.pyro_utils import data_loaders


class VI_BNN(BNN):
    def __init__(self, input_shape, device):
        self.input_size = input_shape[0]*input_shape[1]*input_shape[2]
        self.n_classes = 10
        super(VI_BNN, self).__init__(input_size=self.input_size, device=device)

    def infer_parameters(self, train_loader, lr, n_epochs):
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
                loss = svi.step(inputs=images.to(self.device).view(-1,self.input_size), labels=labels.to(self.device))
                total_loss += loss / len(train_loader.dataset)
                total += labels.size(0)
                pred = self.forward(images.to(device), n_samples=1).mean(0).argmax(-1)
                correct += (pred == labels.argmax(-1).to(self.device)).sum().item()
                accuracy = correct / total * 100

            print(f"[Epoch {i + 1}]\t loss: {total_loss:.2f} \t accuracy: {accuracy:.2f}")
            loss_list.append(total_loss)
            accuracy_list.append(accuracy)

        print("\nlearned params =", list(pyro.get_param_store().get_all_param_names()))
        return {'loss':loss_list, 'accuracy':accuracy_list}


def main(dataset_name, n_inputs, n_samples, n_epochs, lr, device, seed=0):
    random.seed(seed)
    batch_size = 128
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=batch_size, n_inputs=n_inputs)

    ## === infer params ===
    pyro.clear_param_store()
    bayesnn = VI_BNN(input_shape=input_shape, device=device)
    filename = "hidden_vi_" + str(dataset_name) + "_inputs=" + str(n_inputs) + "_lr=" + str(lr) + "_epochs=" + str(
               n_epochs)

    dict = bayesnn.infer_parameters(train_loader=train_loader, n_epochs=n_epochs, lr=lr)
    plot_loss_accuracy(dict, path=RESULTS+"bnn/"+filename+".png")
    bayesnn.save(filename=filename)
    # bayesnn.load(filename=filename, relative_path=RESULTS)

    ## === evaluate ===
    bayesnn.evaluate(test_loader=test_loader)

    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=1, n_inputs=n_inputs)

    expected_loss_gradients(model=bayesnn, n_samples=n_samples, data_loader=test_loader, device="cpu",
                            mode="hidden")


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        n_inputs = int(sys.argv[2])
        n_samples = int(sys.argv[3])
        n_epochs = int(sys.argv[4])
        lr = float(sys.argv[5])
        device = sys.argv[6]

    except IndexError:
        dataset_name = input("\nChoose a dataset: ")
        n_inputs = input("\nChoose the number of samples (type=int): ")
        n_samples = input("\nChoose the number of model samples (type=int): ")
        n_epochs = input("\nSet the number of epochs: ")
        lr = input("\nSet the learning rate: ")
        device = input("\nChoose a device (cpu/gpu): ")

    main(dataset_name,n_inputs, n_samples, n_epochs, lr, device)