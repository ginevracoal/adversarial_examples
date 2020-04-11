import sys
sys.path.append(".")
from directories import *
import argparse
import pyro
import pandas as pd
import matplotlib
from BayesianInference.adversarial_attacks import *
from BayesianInference.loss_gradients import load_loss_gradients, expected_loss_gradients
from BayesianInference.pyro_utils import data_loaders_classes
from tqdm import tqdm

DATA_PATH="../data/exp_loss_gradients/"


def plot_gradients_components_classes(loss_gradients, n_inputs, n_samples_list, dataset):
    matplotlib.rc('font', **{'weight': 'bold', 'size': 12})
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 4), dpi=150, facecolor='w', edgecolor='k')
    
    sns.set_palette("gist_heat", 5)
    # cmap = sns.color_palette("ch:0.8,r=.1,l=.9")
    # sns.set_palette(cmap)
    # sns.set_palette("YlOrRd", 5)

    # len_samp_list, n_classes, n_inputs, n_components == loss_gradients.shape()

    for label in range(10):
        print("\nlabel=", label)
        plot_samples = []
        loss_gradients_components = []
        for samples_idx, n_samples in enumerate(n_samples_list):
            samples_grad_components = np.array(loss_gradients[samples_idx, label]).flatten()
            loss_gradients_components.extend(samples_grad_components)
            plot_samples.extend(np.repeat(n_samples, len(samples_grad_components)))
            
            # print(len(loss_gradients_components), len(plot_samples))

        df = pd.DataFrame(data={"loss_gradients": loss_gradients_components, 
                                "n_samples": plot_samples})
        print(df.describe())

        axis = ax[0, label] if label < 5 else ax[1, label-5]
        
        sns.stripplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, 
                      ax=axis, jitter=0.2, alpha=0.4)

        axis.set_ylabel("")
        axis.set_xlabel("")
        # axis.set_yscale('log')

    fig.text(0.5, 0.01, "Samples involved in the expectations ($w \sim p(w|D)$)", ha='center')
    fig.text(0.03, 0.5, r"Expected Gradients components $\langle\nabla L(x,w)\rangle_{w}$", 
             va='center', rotation='vertical')

    filename = "expLossGrads_inp="+str(n_inputs)+"_"+str(dataset)+".png"
    os.makedirs(os.path.dirname(RESULTS + "plots/"), exist_ok=True)
    fig.savefig(RESULTS + "plots/" + filename)


def main(args):

    # === initialize === #

    # model_idx, dataset = 2, "mnist"
    model_idx, dataset = 5, "fashion_mnist"

    model = hidden_vi_models[model_idx]
    n_inputs = 50
    n_samples_list = [1,10,50,100,500]

    # === compute gradients === #
    
    _, test_loaders, data_format, input_shape = \
        data_loaders_classes(dataset_name=model["dataset"], batch_size=128, n_inputs=n_inputs, 
                             shuffle=False)

    for n_samples in n_samples_list:
        for label, test_loader in enumerate(test_loaders):
            bayesnn = VI_BNN(input_shape=input_shape, device=args.device, 
                             architecture=model["architecture"], activation=model["activation"])
            posterior = bayesnn.load_posterior(posterior_name=model["filename"], 
                                               activation=model["activation"],
                                               relative_path=TRAINED_MODELS, dataset_name=dataset)

            loss_gradients = expected_loss_gradients(posterior=posterior, n_samples=n_samples, 
                             dataset_name=model["dataset"], model_idx=str(label), 
                             data_loader=test_loader, device=args.device, mode="vi")

            del bayesnn, posterior

    # === plot === #

    loss_gradients = []

    for n_samples in n_samples_list:
        samples_gradients = []
        for label in range(10):
            samples_gradients.append(load_loss_gradients(dataset_name=dataset, 
                                  n_inputs=n_inputs, n_samples=n_samples, 
                                  model_idx=str(label), relpath=RESULTS))
        loss_gradients.append(samples_gradients)

    plot_gradients_components_classes(loss_gradients=np.array(loss_gradients), n_inputs=n_inputs,
                                      n_samples_list=n_samples_list, dataset=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')
    main(args=parser.parse_args())