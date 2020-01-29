import sys
sys.path.append(".")
from directories import *

import argparse
import pyro
import pandas as pd
from BayesianInference.adversarial_attacks import *
from BayesianInference.pyro_utils import data_loaders, slice_data_loader
from BayesianInference.hidden_bnn import NN
from BayesianInference.hidden_vi_bnn import VI_BNN


def nn_create_save_data(dataset_name, device):
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=64, n_inputs=60000, shuffle=True)

    accuracy = []
    robustness = []
    epsilon = []
    model_type = []

    count=100
    for eps in [0.1, 0.3, 0.6]:
        for (train_slice, test_slice) in [(i,j) for i in [50, 100, 200] for j in [20, 30, 40]]:

            train = slice_data_loader(train_loader, slice_size=train_slice)
            test = slice_data_loader(test_loader, slice_size=test_slice)

            for epochs in [100, 200]:
                # === initialize class ===
                input_size = input_shape[0] * input_shape[1] * input_shape[2]
                net = NN(input_size=input_size, hidden_size=512, dataset_name=dataset_name, activation="softmax",
                         device=device)
                net.train_classifier(epochs=epochs, lr=0.02, train_loader=train, device=device,
                                     input_size=input_size)

                attack_dict = attack(model=net.model, data_loader=test, epsilon=eps, device=device)

                count += 1
                filename = str(dataset_name) + "_nn_attack_"+str(count)+".pkl"
                save_to_pickle(relative_path=RESULTS + "nn/", filename=filename, data=attack_dict)

                robustness.append(attack_dict["softmax_robustness"])
                accuracy.append(attack_dict["original_accuracy"])
                epsilon.append(attack_dict["epsilon"])
                model_type.append("nn")

    scatterplot_accuracy_robustness(accuracy=accuracy, robustness=robustness, model_type=model_type, epsilon=epsilon)
    return robustness, accuracy, epsilon, model_type


def bnn_create_save_data(dataset_name, device):
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=64, n_inputs=60000, shuffle=True)

    accuracy = []
    robustness = []
    epsilon = []
    model_type = []

    count = 110
    for (train_slice, test_slice) in [(i, j) for i in [50, 100, 200] for j in [20, 30, 40]]:
        train = slice_data_loader(test_loader, slice_size=train_slice)
        test = slice_data_loader(test_loader, slice_size=test_slice)

        for epochs in [50, 80]:
            bayesnn = VI_BNN(input_shape=input_shape, device=device, dataset_name=dataset_name,
                             activation="softmax")
            posterior = bayesnn.infer_parameters(train_loader=train, lr=0.002, n_epochs=epochs)

            for eps in [0.1, 0.3, 0.6]:
                for n_samples in [2]:

                    posterior.evaluate(data_loader=test, n_samples=n_samples)

                    attack_dict = bayesian_attack(model=posterior, data_loader=test, epsilon=eps, device=device,
                                                  n_attack_samples=n_samples, n_pred_samples=n_samples)

                    robustness.append(attack_dict["softmax_robustness"])
                    accuracy.append(attack_dict["original_accuracy"])
                    epsilon.append(eps)
                    model_type.append("bnn")

                    count += 1
                    filename = str(dataset_name) + "_bnn_attack_"+str(count)+".pkl"
                    save_to_pickle(relative_path=RESULTS + "bnn/", filename=filename, data=attack_dict)

    scatterplot_accuracy_robustness(accuracy=accuracy, robustness=robustness, model_type=model_type, epsilon=epsilon)
    return robustness, accuracy, epsilon, model_type


def bnn_create_save_data_pretrained_models(dataset_name, device):
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=64, n_inputs=60000, shuffle=True)

    models_list = [{"idx": 0, "filename": "hidden_vi_mnist_inputs=10000_lr=0.0002_epochs=100", "activation": "softmax",
                    "dataset": "mnist"},
                   {"idx": 1, "filename": "hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=100", "activation": "softmax",
                    "dataset": "mnist"},
                   {"idx": 2, "filename": "hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=11", "activation": "softmax",
                    "dataset": "mnist"}]

    accuracy = []
    robustness = []
    epsilon = []
    model_type = []

    # === load models ===
    count = 82

    for idx in [0]:

        for test_slice in [500]:
            test = slice_data_loader(test_loader, slice_size=test_slice)
            model_dict = models_list[idx]

            bayesnn = VI_BNN(input_shape=input_shape, device=device, dataset_name=model_dict["dataset"],
                             activation=model_dict["activation"])
            posterior = bayesnn.load_posterior(posterior_name=model_dict["filename"], relative_path=TRAINED_MODELS,
                                                   activation=model_dict["activation"])
            for eps in [0.1, 0.3, 0.6]:
                for n_samples in [2]:
                    posterior.evaluate(data_loader=test, n_samples=n_samples)

                    attack_dict = bayesian_attack(model=posterior, data_loader=test, epsilon=eps, device=device,
                                                  n_attack_samples=n_samples, n_pred_samples=n_samples)

                    robustness.append(attack_dict["softmax_robustness"])
                    accuracy.append(attack_dict["original_accuracy"])
                    epsilon.append(eps)
                    model_type.append("bnn")

                    count += 1
                    filename = str(dataset_name) + "_bnn_attack_" + str(count) + ".pkl"
                    save_to_pickle(relative_path=RESULTS + "bnn/", filename=filename, data=attack_dict)

    scatterplot_accuracy_robustness(accuracy=accuracy, robustness=robustness, model_type=model_type, epsilon=epsilon)
    return robustness, accuracy, epsilon, model_type


def create_save_data(dataset_name, device):

    model_type = []
    accuracy = []
    robustness = []
    epsilon = []


    nn_robustness, nn_accuracy, nn_epsilon, nn_model_type = nn_create_save_data(dataset_name, device)
    model_type.extend(nn_model_type)
    robustness.extend(nn_robustness)
    accuracy.extend(nn_accuracy)
    epsilon.extend(nn_epsilon)

    # bnn_robustness, bnn_accuracy, bnn_epsilon, bnn_model_type = bnn_create_save_data_pretrained_models(dataset_name, device)
    bnn_robustness, bnn_accuracy, bnn_epsilon, bnn_model_type = bnn_create_save_data(dataset_name, device)
    model_type.extend(bnn_model_type)
    robustness.extend(bnn_robustness)
    accuracy.extend(bnn_accuracy)
    epsilon.extend(bnn_epsilon)

    return robustness, accuracy, epsilon, model_type


def load_data():

    model_type = []
    accuracy = []
    robustness = []
    epsilon = []

    path = "../data/acc_rob_tradeoff/"

    for file in os.listdir(path+"nn/"):
        if file.endswith(".pkl"):
            dict = load_from_pickle(path=path+"nn/"+file)
            print(dict.keys())
            robustness.append(dict["softmax_robustness"])
            accuracy.append(dict["original_accuracy"])
            epsilon.append(dict["epsilon"])
            model_type.append("nn")

    for file in os.listdir(path+"bnn/"):
        if file.endswith(".pkl"):
            dict = load_from_pickle(path=path+"bnn/"+file)
            print(dict.keys())
            robustness.append(dict["softmax_robustness"])
            accuracy.append(dict["original_accuracy"])
            epsilon.append(dict["epsilon"])
            model_type.append("bnn")

    path = "../data/exp_loss_gradients/"
    for file in os.listdir(path):
        if file.endswith(".pkl"):
            dict = load_from_pickle(path=path+file)
            print(dict.keys())
            robustness.append(dict["softmax_robustness"])
            accuracy.append(dict["original_accuracy"])
            epsilon.append(dict["epsilon"])
            model_type.append("bnn")

    return robustness, accuracy, epsilon, model_type


def scatterplot_accuracy_robustness(accuracy, robustness, model_type, epsilon):
    """
    Scatterplot of accuracy (x axis) vs robustness (y axis) with categorical model_type.
    """
    sns.set()
    plt.subplots(figsize=(10, 6), dpi=150)
    sns.set_palette("YlGnBu",2)

    # size = ["$%s$" % x for x in epsilon]
    df = pd.DataFrame(data={"accuracy":accuracy,"robustness":robustness,"model":model_type,"epsilon":epsilon})
    print(df.head())

    for _ in range(len(accuracy)):
        sns.scatterplot(data=df, x="accuracy", y="robustness", hue="model", style="epsilon", alpha=0.8, linewidth=0.1)

    plt.xlabel('Test accuracy (%) ')
    plt.ylabel('Softmax robustness ($l_\infty$)')
    plt.title("Accuracy vs robustness for VI BNNs on MNIST")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower left')

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS + "models=" +str(len(accuracy))+ "_scatterplot_accuracy_robustness.png")


def main(args):

    # robustness, accuracy, epsilon, model_type = create_save_data(dataset_name=args.dataset, device=args.device)
    robustness, accuracy, epsilon, model_type = load_data()
    scatterplot_accuracy_robustness(accuracy=accuracy, robustness=robustness, model_type=model_type, epsilon=epsilon)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')

    main(args=parser.parse_args())
