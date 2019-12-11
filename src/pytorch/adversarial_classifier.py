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
from torch.utils.data import DataLoader
from utils import *
from pytorch.nets import torch_net
from pytorch.sgd import SGD
import time


RESULTS = "../../results/"+str(time.strftime('%Y-%m-%d'))+"/"
DATA_PATH = "../data/"
DATASETS = "mnist, cifar"


def set_device(device_name):
    if device_name == "gpu":
        return torch.device("gpu")
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

    def set_optimizer(self):
        return NotImplementedError

    def train(self, x_train, y_train, val_ratio=0.2, lr=0.01, epochs=10, device="cpu"):
        model = self.net

        def validation_split(x_train, y_train, val_ratio):
            split_idx = int(len(x_train)*(1-val_ratio))
            train_data = list(zip(x_train[:split_idx,:],y_train[:split_idx,:]))
            val_data = list(zip(x_train[split_idx:,:],y_train[split_idx:,:]))
            dataloader = {'training':DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=2),
                          'validation':DataLoader(dataset=val_data, batch_size=128, shuffle=True, num_workers=2)}
            dataset_sizes = {'training':split_idx,'validation':len(x_train)-split_idx}
            return dataloader, dataset_sizes

        dataloader, dataset_sizes = validation_split(x_train, y_train, val_ratio)

        start = time.time()
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            # print('-' * 10)

            # todo: add early stopping and tensorboard callbacks
            for phase in ['training', 'validation']:
                if phase == 'training':
                    # scheduler.step() # changes the lr
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                optimizer = optim.SGD(model.parameters(), lr=lr)  # torch implementation
                # optimizer = optim.Adam(model.parameters(), lr=0.001)
                # print(list(model.parameters()))
                # optimizer = SGD(params=list(model.parameters()), lr=0.001) # my implementation

                running_loss = 0.0
                correct_preds = 0.0
                # for i in range(self.n_samples):
                for i, data in enumerate(dataloader[phase], 0):
                    # load inputs and labels
                    inputs, one_hot_labels = data
                    labels = np.argmax(one_hot_labels, axis=1)
                    # print(inputs.shape, one_hot_labels.shape)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'training'):
                        # forward
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = nn.CrossEntropyLoss()(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'training':
                            loss.backward()
                            optimizer.step()

                    # print statistics
                    running_loss += loss.item() * inputs.size(0)
                    correct_preds += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = correct_preds / dataset_sizes[phase]
            print('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        time_elapsed = time.time() - start
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.net = model
        self.trained = True

    def save_classifier(self, relative_path, filename=None):
        # todo: come sta aggiornando self.net()? in quale metodo?
        if filename is None:
            filename = self.classifier_name
        os.makedirs(os.path.dirname(relative_path), exist_ok=True)
        torch.save(self.net.state_dict(), relative_path+filename+".h5")

    def evaluate(self, x_test, y_test, device="cpu"):
        model = self.net
        model.eval()

        test_data = list(zip(x_test,y_test))
        testloader = DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=2)

        # x = torch.tensor(x)
        if self.trained:
            with torch.no_grad():
                # for i, data in enumerate(dataloader[phase], 0):
                #     # load inputs and labels
                #     inputs, one_hot_labels = data
                #     labels = np.argmax(one_hot_labels, axis=1)

                for idx, (inputs, one_hot_labels) in enumerate(testloader):
                    outputs = model.forward(inputs)
                    labels = np.argmax(one_hot_labels, axis=1)
                    _, predicted = outputs.max(dim=1)

                    # check predictions
                    if idx == 0:
                        print(predicted)  # the predicted class
                        print(torch.exp(_))  # the predicted probability
                    equals = predicted == labels.data

                accuracy = equals.float().mean()
                print("Accuracy: %.2f%%" % (accuracy * 100))
            # print(classification_report(y_true, y_pred, labels=list(range(self.num_classes))))
            # return accuracy
        else:
            raise AttributeError("Train your classifier before the evaluation.")


def main(dataset_name, test, device, seed):

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                           test=test)
    model = AdversarialClassifier(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                  dataset_name=dataset_name, test=test)
    model.train(x_train, y_train)
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