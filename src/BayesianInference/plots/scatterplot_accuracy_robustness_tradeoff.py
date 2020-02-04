import sys
sys.path.append(".")
from directories import *

import pandas as pd
from BayesianInference.adversarial_attacks import *
from BayesianInference.pyro_utils import data_loaders, slice_data_loader
from BayesianInference.hidden_bnn import NN
from BayesianInference.hidden_vi_bnn import VI_BNN
import itertools

REL_PATH = "scatterplot_attacks/"
DATA_PATH = "../data/"

EPS=0.25
ATTACK_SAMPLES=50
PRED_SAMPLES=3
TEST_IMAGES=[71, 93, 100]
ARCHIT = ["fully_connected", "fully_connected_2"]
HIDDEN = [128, 256, 512]
EPOCHS = [10, 20, 30]
LR = [0.02, 0.002, 0.0002]
ACTIVATION = ["leaky_relu", "tanh"]#, "sigmoid"]
INPUTS = [30000, 60000]


def _train_and_attack(model_idx, architecture, hidden_size, epochs, lr, activation, test_images,
           dataset_name, n_inputs, model_type, device, n_pred_samples, n_attack_samples, eps):

    train_loader, test_loader, _, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=128, n_inputs=n_inputs, shuffle=True)
    input_size = input_shape[0] * input_shape[1] * input_shape[2]
    test = slice_data_loader(test_loader, slice_size=test_images)

    print(f"\narchitecture = {architecture} - activation = {activation} - hidden_size = {hidden_size}"
          f" - epochs = {epochs}")

    if model_type == "nn":
        net = NN(input_size=input_size, hidden_size=hidden_size, architecture=architecture,
                 activation=activation, device=device)
        net.train_classifier(epochs=epochs, lr=lr, train_loader=train_loader, device=device, input_size=input_size,
                             dataset_name=dataset_name)
        attack(model=net.model, data_loader=test, epsilon=eps, device=device,
               save_dir=RESULTS+REL_PATH+"nn/attacks/",  method="fgsm",
               dataset_name=dataset_name, model_idx=model_idx)
        del net

    elif model_type == "bnn":
        bayesnn = VI_BNN(input_shape=input_shape, device=device, architecture=architecture,
                         activation=activation, hidden_size=hidden_size)
        posterior = bayesnn.infer_parameters(train_loader=train_loader, lr=lr, n_epochs=epochs,
                                             dataset_name=dataset_name)
        bayesian_attack(model=posterior, n_attack_samples=n_attack_samples,
                        save_dir=RESULTS+REL_PATH+"bnn/attacks/", method="fgsm",
                        data_loader=test, n_pred_samples=n_pred_samples,
                        dataset_name=dataset_name, epsilon=eps, device=device,
                        model_idx=model_idx)
        del bayesnn
    else:
        raise AssertionError("wrong model_type.")

def parallel_grid_search(dataset_name, model_type, n_pred_samples, n_attack_samples, eps, n_jobs):

    print("\n == Grid search training == ")
    from joblib import Parallel, delayed

    combinations = list(itertools.product(ARCHIT, HIDDEN, EPOCHS, LR, ACTIVATION, INPUTS, TEST_IMAGES))
    Parallel(n_jobs=n_jobs)(
        delayed(_train_and_attack)(idx, architecture, hidden_size, epochs, lr, activation, test_images,
                      dataset_name, n_inputs, model_type, "cpu", n_pred_samples, n_attack_samples, eps)
        for idx, (architecture, hidden_size, epochs, lr, activation, n_inputs, test_images) in enumerate(combinations))

    scatterplot_dict = build_scatterplot_dict(dataset_name, model_type, n_attack_samples=n_attack_samples,
                        n_pred_samples=n_pred_samples, eps=eps, relative_path=RESULTS)
    return scatterplot_dict


def serial_grid_search(dataset_name, model_type, n_pred_samples, n_attack_samples, eps, device):

    print("\n == Grid search training == ")

    combinations = list(itertools.product(ARCHIT, HIDDEN, EPOCHS, LR, ACTIVATION, INPUTS, TEST_IMAGES))

    for idx, (architecture, hidden_size, epochs, lr, activation, n_inputs, test_images) in enumerate(combinations):
        _train_and_attack(model_idx=idx, architecture=architecture, hidden_size=hidden_size,
                          epochs=epochs, lr=lr, activation=activation, dataset_name=dataset_name, n_inputs=n_inputs,
                          model_type=model_type, device=device, n_pred_samples=n_pred_samples,
                          n_attack_samples=n_attack_samples, eps=eps, test_images=test_images)

    scatterplot_dict = build_scatterplot_dict(dataset_name, model_type, n_attack_samples=n_attack_samples,
                  n_pred_samples=n_pred_samples, eps=eps, relative_path=RESULTS)
    return scatterplot_dict


def build_scatterplot_dict(dataset_name, model_type, n_attack_samples=ATTACK_SAMPLES,
                 n_pred_samples=PRED_SAMPLES, eps=EPS, relative_path=RESULTS):

    print("\n == Build scatterplot dict == ")

    original_accuracy = []
    adversarial_accuracy = []
    robustness = []
    model_type_list = []

    attacks_dir = relative_path+REL_PATH+str(dataset_name)+"/"+str(model_type)+"/attacks/"
    for file in os.listdir(attacks_dir):
        if file.endswith(".pkl"):
            attack_dict = load_from_pickle(path=attacks_dir+file)
            robustness.append(attack_dict["softmax_robustness"])
            original_accuracy.append(attack_dict["original_accuracy"])
            adversarial_accuracy.append(attack_dict["adversarial_accuracy"])
            model_type_list.append(str(model_type))

    scatterplot_dict = {"softmax_robustness":robustness, "original_accuracy":original_accuracy,
                        "adversarial_accuracy":adversarial_accuracy, "model_type":model_type_list}
    print(scatterplot_dict)
    filename = str(dataset_name)+"_"+str(model_type)+"_models="+str(len(robustness))+"_attackSamp="\
               +str(n_attack_samples)+"_predSamp="+str(n_pred_samples)+"_eps="+str(eps)+"_scatterplot.pkl"
    save_to_pickle(data=scatterplot_dict, relative_path=RESULTS+REL_PATH+str(dataset_name)+"/", filename=filename)
    return scatterplot_dict


def load_data(dataset_name, relative_path="../data/"):

    model_type = []
    accuracy = []
    robustness = []

    for model in ["nn", "bnn"]:
        path = relative_path+REL_PATH+str(dataset_name)+"/"+str(model)+"/"
        for file in os.listdir(path):
            if file.endswith(".pkl"):
                dict = load_from_pickle(path=path+file)
                robustness.extend(dict["softmax_robustness"])
                accuracy.extend(dict["original_accuracy"])
                model_type.extend(dict["model_type"])

    dict = {"softmax_robustness":robustness, "model_type":model_type, "original_accuracy":accuracy}
    print("\nLoaded dict:\n", dict)
    return dict


def scatterplot_accuracy_robustness(scatterplot_dict, dataset_name):
    """
    Scatterplot of accuracy (x axis) vs robustness (y axis) with categorical model_type.
    """
    sns.set()
    plt.subplots(figsize=(8, 6), dpi=150)
    sns.set_palette("YlGnBu_d", 2)

    df = pd.DataFrame(data={"accuracy":scatterplot_dict["original_accuracy"],
                            "robustness":scatterplot_dict["softmax_robustness"],
                            "model":scatterplot_dict["model_type"]})
    df = df[df['accuracy'] > 60.0]
    print(df.head())
    print(df.describe())
    n_models = len(df.index)

    g = sns.scatterplot(data=df, x="accuracy", y="robustness", hue="model", alpha=0.8, linewidth=0.1)
    g.set(xlim=(60,None))

    plt.xlabel('Test accuracy (%) ', fontsize=11)
    plt.ylabel('Softmax robustness ($l_\infty$)', fontsize=11)
    plt.title("Accuracy vs robustness for VI BNNs on MNIST")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower left')

    os.makedirs(os.path.dirname(RESULTS+REL_PATH), exist_ok=True)
    plt.savefig(RESULTS+REL_PATH+str(dataset_name)+"/"
                +str(dataset_name)+"_models="+str(n_models)+ "_scatterplot_accuracy_robustness.png")


def main(args):

    for model_type in ["bnn"]:
        # scatterplot_dict = parallel_grid_search(dataset_name=args.dataset, model_type=model_type,
        #                                         n_attack_samples=ATTACK_SAMPLES, eps=EPS,
        #                                         n_pred_samples=PRED_SAMPLES, n_jobs=12)
        scatterplot_dict = serial_grid_search(dataset_name=args.dataset, device=args.device, model_type=model_type,
                                                n_attack_samples=ATTACK_SAMPLES, eps=EPS,
                                                n_pred_samples=PRED_SAMPLES)

    # build_scatterplot_dict(args.dataset, model_type=model_type, n_attack_samples=ATTACK_SAMPLES,
    #                        n_pred_samples=PRED_SAMPLES, eps=EPS, relative_path=RESULTS)

    # scatterplot_dict = load_data(relative_path=DATA_PATH, dataset_name=args.dataset)

    scatterplot_accuracy_robustness(scatterplot_dict=scatterplot_dict, dataset_name=args.dataset)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--dataset", default='mnist', type=str, help='use "mnist" or "fashion_mnist".')
    main(args=parser.parse_args())
