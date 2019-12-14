""" Torch implementation of:
- Neural network architectures for MNIST and CIFAR datasets
- SGD optimization
- Adversarial classifier class, implementing basic adversarial methods
"""

import sys
sys.path.append('../')

import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from pytorch.nets import torch_net
from pytorch.sgd import SGD, BayesianSGD
import time
import random
import copy
from torch.utils.data.dataset import random_split


RESULTS = "../../results/"+str(time.strftime('%Y-%m-%d'))+"/"
DATA_PATH = "../data/"
DATASETS = "mnist, cifar"


def validation_split(x_train, y_train, val_ratio, batch_size):
    split_idx = int(len(x_train) * (1 - val_ratio))
    x_tensor = torch.from_numpy(x_train).float()
    y_tensor = torch.from_numpy(y_train).float()
    dataset = TensorDataset(x_tensor, y_tensor)
    train_dataset, val_dataset = random_split(dataset, [split_idx, len(x_train) - split_idx])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
    return train_loader, val_loader

def set_device(device_name):
    if device_name == "gpu":
        return torch.device("cuda")
    elif device_name == "cpu":
        return torch.device("cpu")
    else:
        raise AssertionError("Wrong device name.")


class SGDClassifier(object):
    def __init__(self, input_shape, num_classes, data_format, dataset_name, test):
        self.net = torch_net(dataset_name=dataset_name, input_shape=input_shape, data_format=data_format)
        self.batch_size = 128
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.data_format = data_format
        self.dataset_name = dataset_name
        self.test = test
        self.name = dataset_name + str("_sgd")

    def set_optimizer(self, lr):
        return SGD(params=list(self.net.parameters()), lr=lr)

    def train_step(self, inputs, labels, optimizer_params):
        model = self.net
        optimizer = self.optimizer
        model.train()  # train mode
        outputs = model(inputs)  # make predictions
        # print("check updates: outputs[0]=",outputs[0],", labels:",labels)
        loss = self.loss_fn(outputs, labels)  # compute loss
        loss.backward()  # compute gradients
        optimizer.step(optimizer_params)  # update parameters
        optimizer.zero_grad()  # put gradients to zero
        return loss.item()

    def train_epoch(self, model, train_loader, val_loader, device, optimizer_params):
        losses = []
        val_losses = []

        # === training === #
        train_loss = 0.0
        count_predictions = 0
        correct_predictions = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            labels = onehot_to_labels(y_batch).to(device)
            train_loss = self.train_step(x_batch, labels, optimizer_params)
            losses.append(train_loss)
            predictions = onehot_to_labels(model(x_batch))
            count_predictions += y_batch.size(0)
            correct_predictions += (predictions == labels).sum().item()
        train_acc = 100 * correct_predictions / count_predictions
        print('Training loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc), end='   ')

        # === validation === #
        with torch.no_grad():  # temporarily set all the requires_grad flag to false
            model.eval()
            val_loss = 0.0
            count_predictions = 0
            correct_predictions = 0
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                labels = onehot_to_labels(y_val).to(device)

                outputs = model(x_val)
                val_loss = self.loss_fn(outputs, labels)
                val_losses.append(val_loss.item())

                predictions = onehot_to_labels(model(x_val))
                count_predictions += labels.size(0)
                correct_predictions += (predictions == labels).sum().item()
            val_acc = 100 * correct_predictions / count_predictions
            print('Validation loss: {:.4f}, acc: {:.4f}'.format(val_loss, val_acc), end='   ')

    def train(self, x_train, y_train, val_ratio=0.2, lr=0.01, epochs=100, device="cpu"):
        model = self.net
        print(model)

        if device == "gpu":
            model.cuda()
        device = set_device(device_name=device)

        self.lr = lr
        self.optimizer = self.set_optimizer(lr=lr)

        train_loader, val_loader = validation_split(x_train, y_train, val_ratio, self.batch_size)

        start = time.time()

        n_training_samples = len(train_loader)
        optimizer_params = {'n_training_samples':n_training_samples}
        for epoch in range(epochs):
            print("\n", '-' * 10)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            self.train_epoch(model=model, train_loader=train_loader, val_loader=val_loader, device=device,
                             optimizer_params=optimizer_params.update({'epoch':epoch}))

        time_elapsed = time.time() - start
        print('\n\nTraining time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.net = model
        self.trained = True
        # print(model.state_dict())

    def save_classifier(self, relative_path, filename=None):
        if filename is None:
            filename = self.name
        os.makedirs(os.path.dirname(relative_path), exist_ok=True)
        torch.save(self.net.state_dict(), relative_path+filename+".h5")

    def evaluate(self, x_test, y_test, device="cpu"):
        model = self.net
        model.eval()
        if device == "gpu":
            model.cuda()

        set_device(device_name=device)
        test_data = list(zip(x_test,y_test))
        test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True, num_workers=2)

        if self.trained:
            with torch.no_grad():
                correct_predictions = 0
                for idx, (x_test, y_test) in enumerate(test_loader):
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                    outputs = model(x_test)
                    labels = onehot_to_labels(y_test)
                    values, predictions = outputs.max(dim=1)
                    # check predictions
                    if idx == 0:
                        print(labels)
                        print("predictions: ", predictions)  # the predicted class
                        print("probabilities: ", values)  # the predicted probability
                    correct_predictions += (predictions == labels).sum()

                accuracy = correct_predictions / len(predictions)
                print("Accuracy: %.2f%%" % (accuracy * 100))
            return accuracy
        else:
            raise AttributeError("Train your classifier before the evaluation.")


class BayesianSGDClassifier(SGDClassifier):
    def __init__(self, input_shape, num_classes, data_format, dataset_name, test):
        super(BayesianSGDClassifier, self).__init__(input_shape, num_classes, data_format, dataset_name, test)
        self.batch_size = 1000
        self.classifier_name = dataset_name + str("_bayesian_sgd")

    def set_optimizer(self, lr):
        return BayesianSGD(params=list(self.net.parameters()), loss_fn=self.loss_fn)

    def train_step(self, inputs, labels, optimizer_params):
        model = self.net
        optimizer = self.optimizer
        model.train()  # train mode
        outputs = model(inputs)  # make predictions
        loss = self.loss_fn(outputs, labels)  # compute loss
        loss.backward()  # compute gradients
        optimizer.step(outputs=outputs, labels=labels, optimizer_params=optimizer_params)  # update parameters
        optimizer.zero_grad()  # put gradients to zero
        return loss.item()


def main(dataset_name, test, device, seed):
    random.seed(seed)

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                           test=test)

    model = BayesianSGDClassifier(
    # model = SGDClassifier(
        input_shape=input_shape, num_classes=num_classes, data_format=data_format, dataset_name=dataset_name, test=test)

    lr = 0.01
    epochs = 100
    model.train(x_train, y_train, device=device, lr=lr, epochs=epochs)
    model.save_classifier(relative_path=RESULTS, filename=model.name+"_lr="+str(lr)+"_epochs="+str(epochs)+"_"+str(seed))
    model.evaluate(x_test=x_test, y_test=y_test)


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        device = sys.argv[3]
        seed = int(sys.argv[4])

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        device = input("\nChoose a device (cpu/gpu): ")
        seed = input("\nSet a training seed (type=int): ")

    main(dataset_name=dataset_name, test=test, device=device, seed=seed)