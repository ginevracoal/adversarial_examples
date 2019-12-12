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
from pytorch.sgd import SGD
import time
import random
import copy
from torch.utils.data.dataset import random_split


RESULTS = "../../results/"+str(time.strftime('%Y-%m-%d'))+"/"
DATA_PATH = "../data/"
DATASETS = "mnist, cifar"


def set_device(device_name):
    if device_name == "gpu":
        return torch.device("cuda")
    elif device_name == "cpu":
        return torch.device("cpu")
    else:
        raise AssertionError("Wrong device name.")


class AdversarialClassifier(object):
    def __init__(self, input_shape, num_classes, data_format, dataset_name, test, library="cleverhans"):
        self.net = torch_net(dataset_name=dataset_name, input_shape=input_shape, data_format=data_format)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.data_format = data_format
        self.dataset_name = dataset_name
        self.test = test
        self.library = library
        self.classifier_name = dataset_name + str("_sgd_classifier")

    def set_optimizer(self, model, lr):
        return SGD(params=list(model.parameters()), lr=lr)

    def train(self, x_train, y_train, val_ratio=0.2, lr=0.01, epochs=100, device="cpu"):
        # y_train = np.argmax(y_train, axis=1) # one_hot encoding to labels
        model = self.net
        if device == "gpu":
            model.cuda()
        device = set_device(device_name=device)
        # print(model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = self.set_optimizer(model=self.net, lr=lr)

        def validation_split(x_train, y_train, val_ratio):
            split_idx = int(len(x_train) * (1 - val_ratio))
            x_tensor = torch.from_numpy(x_train).float()
            y_tensor = torch.from_numpy(y_train).float()
            dataset = TensorDataset(x_tensor, y_tensor)
            train_dataset, val_dataset = random_split(dataset, [split_idx, len(x_train)-split_idx])

            train_loader = DataLoader(dataset=train_dataset, batch_size=128)
            val_loader = DataLoader(dataset=val_dataset, batch_size=128)
            return train_loader, val_loader

        def train_step(inputs, labels):
            model.train()  # train mode
            outputs = model(inputs)  # make predictions
            # print("check updates: outputs[0]=",outputs[0],", labels:",labels)
            loss = loss_fn(outputs, labels)  # compute loss
            loss.backward()  # compute gradients
            optimizer.step()  # update parameters
            optimizer.zero_grad()  # put gradients to zero
            return loss.item()

        train_loader, val_loader = validation_split(x_train, y_train, val_ratio)

        start = time.time()
        losses = []
        val_losses = []
        for epoch in range(epochs):
            print("\n",'-' * 10)
            print('Epoch {}/{}'.format(epoch, epochs - 1))

            # === training ===
            train_loss = 0.0
            count_predictions = 0
            correct_predictions = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                labels = onehot_to_labels(y_batch).to(device)
                train_loss = train_step(x_batch, labels)
                losses.append(train_loss)
                predictions = onehot_to_labels(model(x_batch))
                count_predictions += labels.size(0)
                correct_predictions += (predictions == labels).sum().item()
            train_acc = 100 * correct_predictions / count_predictions
            print('Training loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc), end='   ')

            # === validation ===
            with torch.no_grad():
                val_loss = 0.0
                count_predictions = 0
                correct_predictions = 0
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    labels = onehot_to_labels(y_val).to(device)

                    model.eval()

                    outputs = model(x_val)
                    val_loss = loss_fn(outputs, labels)
                    val_losses.append(val_loss.item())

                    predictions = onehot_to_labels(model(x_val))
                    count_predictions += labels.size(0)
                    correct_predictions += (predictions == labels).sum().item()
                val_acc = 100 * correct_predictions / count_predictions
                print('Validation loss: {:.4f}, acc: {:.4f}'.format(val_loss, val_acc), end='   ')

        time_elapsed = time.time() - start
        print('\n\nTraining time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.net = model
        self.trained = True
        # print(model.state_dict())

    def save_classifier(self, relative_path, filename=None):
        if filename is None:
            filename = self.classifier_name
        os.makedirs(os.path.dirname(relative_path), exist_ok=True)
        torch.save(self.net.state_dict(), relative_path+filename+".h5")

    def evaluate(self, x_test, y_test, device="cpu"):
        model = self.net
        if device == "gpu":
            model.cuda()

        model.eval()
        set_device(device_name=device)
        test_data = list(zip(x_test,y_test))
        testloader = DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=2)

        if self.trained:
            with torch.no_grad():
                correct_predictions = 0
                for idx, (x_test, y_test) in enumerate(testloader):
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


def main(dataset_name, test, device, seed):
    random.seed(seed)

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                           test=test)
    model = AdversarialClassifier(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                  dataset_name=dataset_name, test=test)
    model.train(x_train, y_train, device=device, lr=0.01, epochs=300)
    model.save_classifier(relative_path=RESULTS)
    model.evaluate(x_test=x_test, y_test=y_test)
    # for attack in ['fgsm']:
    #     # x_test_adv = model.generate_adversaries(x=x_test, y=y_test, attack=attack, seed=seed, eps=eps)
    #     # model.save_adversaries(data=x_test_adv, attack=attack, seed=seed, eps=eps)
    #     x_test_adv = model.load_adversaries(attack=attack, relative_path=DATA_PATH, seed=seed)
    #     model.evaluate(x=x_test_adv, y=y_test)


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