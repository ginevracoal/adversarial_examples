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
import itertools


DATA_PATH = "../data/rob_acc_tradeoff/"

EPS=0.25
ATTACK_SAMPLES=100
PRED_SAMPLES=100
TEST_IMAGES=600
START_MODEL_IDX=100


def _train_and_attack(model_idx, architecture, hidden_size, epochs, lr, activation,
                      dataset_name, n_inputs, model_type, device):
    n_pred_samples=PRED_SAMPLES
    n_attack_samples=ATTACK_SAMPLES
    eps=EPS

    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=128, n_inputs=n_inputs, shuffle=True)

    test = slice_data_loader(test_loader, slice_size=TEST_IMAGES)
    input_size = input_shape[0] * input_shape[1] * input_shape[2]

    print(f"\narchitecture = {architecture} - activation = {activation} - hidden_size = {hidden_size}"
          f" - epochs = {epochs}")

    if model_type == "nn":
        net = NN(input_size=input_size, hidden_size=hidden_size, architecture=architecture,
                 activation=activation, device=device)
        net.train_classifier(epochs=epochs, lr=lr, train_loader=train_loader, device=device,
                             input_size=input_size)
        attack(model=net.model, data_loader=test, epsilon=eps, device=device,
               dataset_name=dataset_name, model_idx=START_MODEL_IDX+model_idx)
        del net

    elif model_type == "bnn":
        bayesnn = VI_BNN(input_shape=input_shape, device=device, architecture=architecture,
                         activation=activation, hidden_size=hidden_size)
        posterior = bayesnn.infer_parameters(train_loader=train_loader, lr=lr, n_epochs=epochs,
                                             dataset_name=dataset_name)
        bayesian_attack(model=posterior, n_attack_samples=n_attack_samples,
                        data_loader=test_loader, n_pred_samples=n_pred_samples,
                        dataset_name=dataset_name, epsilon=eps, device=device,
                        model_idx=START_MODEL_IDX+model_idx)
        del bayesnn, posterior
    else:
        raise AssertionError("wrong model_type.")


def parallel_grid_search_training(dataset_name, model_type, n_inputs=60000):
    from joblib import Parallel, delayed

    architecture = ["fully_connected", "fully_connected_2", "convolutional"]
    hidden_size = [128, 256, 512]
    epochs = [10, 15, 20]
    lr = [0.02, 0.002, 0.0002, 0.00002]
    activation = ["leaky_relu", "tanh", "sigmoid"]
    combinations = list(itertools.product(architecture, hidden_size, epochs, lr, activation))

    Parallel(n_jobs=20)(
        delayed(_train_and_attack)(idx, architecture, hidden_size, epochs, lr, activation,
                      dataset_name, n_inputs, model_type, "cpu")
        for idx, (architecture, hidden_size, epochs, lr, activation) in enumerate(combinations))


def grid_search_training(dataset_name, model_type, device="cuda", n_inputs=60000):

    model_idx = START_MODEL_IDX
    for architecture in ["fully_connected","fully_connected_2","convolutional"]:
        for hidden_size in [128,256,512]:
            for epochs in [10,15,20]:
                for lr in [0.02, 0.002, 0.0002, 0.00002]:
                    for activation in ["leaky_relu", "tanh", "sigmoid"]:
                        _train_and_attack(model_idx, architecture, hidden_size, epochs, lr, activation,
                                          dataset_name, n_inputs, model_type, device)
                        model_idx += 1

def prepare_data(dataset_name, n_models, n_attack_samples=ATTACK_SAMPLES, n_pred_samples=PRED_SAMPLES, eps=EPS):

    original_accuracy = []
    adversarial_accuracy = []
    robustness = []
    model_type = []

    # todo qua mettere gli indici giusti, controllare che stia caricando tutto
    filename_idxs = range(100, n_models+100)

    for idx in filename_idxs:
        filename = str(dataset_name)+"_nn_inputs="+str(TEST_IMAGES)+"_eps="+str(eps)+"_attack_"+str(idx)+".pkl"
        attack_dict = load_from_pickle(DATA_PATH+"attacks/"+filename)
        robustness.append(attack_dict["softmax_robustness"])
        original_accuracy.append(attack_dict["original_accuracy"])
        adversarial_accuracy.append(attack_dict["adversarial_accuracy"])
        model_type.append("nn")

    # for idx in filename_idxs:
    #     filename = str(dataset_name)+"_bnn_inputs=" + str(TEST_IMAGES) + "_attackSamp=" + str(n_attack_samples) \
    #                +"_predSamp=" +str(n_pred_samples)+ "_eps=" + str(eps) + "_attack_" + str(idx) + ".pkl"
    #     attack_dict = load_from_pickle(DATA_PATH+"attacks/"+filename)
    #     robustness.append(attack_dict["softmax_robustness"])
    #     original_accuracy.append(attack_dict["original_accuracy"])
    #     adversarial_accuracy.append(attack_dict["adversarial_accuracy"])
    #     model_type.append("bnn")

    scatterplot_dict = {"softmax_robustness":robustness, "original_accuracy":original_accuracy,
                        "adversarial_accuracy":adversarial_accuracy, "model_type":model_type}
    filename = str(dataset_name)+"_models="+str(n_models)+"_attackSamp="+str(n_attack_samples)+\
               "_predSamp="+str(n_pred_samples)+"_eps="+str(eps)+"_scatterplot.pkl"
    save_to_pickle(data=scatterplot_dict, relative_path=RESULTS, filename=filename)
    return scatterplot_dict


def load_data(dataset_name, n_models, n_attack_samples=ATTACK_SAMPLES, n_pred_samples=PRED_SAMPLES,
              eps=EPS, relative_path=DATA_PATH):
    filename = str(dataset_name)+"_models="+str(n_models)+"_attackSamp="+str(n_attack_samples)+\
               "_predSamp="+str(n_pred_samples)+"_eps="+str(eps)+"_scatterplot.pkl"
    return load_from_pickle(relative_path+filename)

# todo remove, deprecated
# def load_old_data():
#
#     model_type = []
#     accuracy = []
#     robustness = []
#     epsilon = []
#
#     path = "../data/acc_rob_tradeoff/"
#
#     for file in os.listdir(path+"nn/"):
#         if file.endswith(".pkl"):
#             dict = load_from_pickle(path=path+"nn/"+file)
#             print(dict.keys())
#             robustness.append(dict["softmax_robustness"])
#             accuracy.append(dict["original_accuracy"])
#             epsilon.append(dict["epsilon"])
#             model_type.append("nn")
#
#     for file in os.listdir(path+"bnn/"):
#         if file.endswith(".pkl"):
#             dict = load_from_pickle(path=path+"bnn/"+file)
#             print(dict.keys())
#             robustness.append(dict["softmax_robustness"])
#             accuracy.append(dict["original_accuracy"])
#             epsilon.append(dict["epsilon"])
#             model_type.append("bnn")
#
#     path = "../data/exp_loss_gradients/"
#     for file in os.listdir(path):
#         if file.endswith(".pkl"):
#             dict = load_from_pickle(path=path+file)
#             print(dict.keys())
#             robustness.append(dict["softmax_robustness"])
#             accuracy.append(dict["original_accuracy"])
#             epsilon.append(dict["epsilon"])
#             model_type.append("bnn")
#
#     return robustness, accuracy, epsilon, model_type


def scatterplot_accuracy_robustness(scatterplot_dict):
    """
    Scatterplot of accuracy (x axis) vs robustness (y axis) with categorical model_type.
    """
    sns.set()
    plt.subplots(figsize=(8, 6), dpi=150)
    sns.set_palette("YlGnBu_d")#,2)

    # size = ["$%s$" % x for x in epsilon]
    n_models = len(scatterplot_dict["original_accuracy"])
    df = pd.DataFrame(data={"accuracy":scatterplot_dict["original_accuracy"],
                            "robustness":scatterplot_dict["softmax_robustness"],
                            "model":scatterplot_dict["model_type"]}) #,"epsilon":epsilon})
    print(df.head())

    for _ in range(n_models):
        sns.scatterplot(data=df, x="accuracy", y="robustness", hue="model", alpha=0.8, linewidth=0.1) #style="epsilon",

    plt.xlabel('Test accuracy (%) ', fontsize=11)
    plt.ylabel('Softmax robustness ($l_\infty$)', fontsize=11)
    plt.title("Accuracy vs robustness for VI BNNs on MNIST")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower left')

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS + "models=" +str(n_models)+ "_scatterplot_accuracy_robustness.png")


def main(args):

    # parallel_grid_search_training(dataset_name=args.dataset, model_type="nn", n_inputs=args.inputs)
    # parallel_grid_search_training(dataset_name=args.dataset, model_type="bnn", n_inputs=args.inputs)
    # grid_search_training(dataset_name=args.dataset, device=args.device, model_type="nn", n_inputs=args.inputs)
    # grid_search_training(dataset_name=args.dataset, device=args.device, model_type="bnn", n_inputs=args.inputs)

    # scatterplot_dict = prepare_data(dataset_name=args.dataset, n_models=139)
    scatterplot_dict = prepare_data(dataset_name=args.dataset, n_models=53)
    # scatterplot_dict = load_data(dataset_name=args.dataset, n_models=162, relative_path=DATA_PATH)
    scatterplot_accuracy_robustness(scatterplot_dict)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--inputs", default=10, type=int)

    main(args=parser.parse_args())
