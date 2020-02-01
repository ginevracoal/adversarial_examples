import sys
sys.path.append(".")
from directories import *
import argparse
import pyro
import pandas as pd
import matplotlib
from BayesianInference.adversarial_attacks import *
from BayesianInference.loss_gradients import load_loss_gradients, categorical_loss_gradients_norms

DATA_PATH="../data/exp_loss_gradients/"


def plot_exp_loss_gradients_norms(loss_gradients, n_inputs, n_samples_list, dataset_name, model_idx):
    plot_loss_gradients = []
    plot_samples = []

    for samples_idx, n_samples in enumerate(n_samples_list):
        print("\n\nsamples = ", n_samples, end="\t")
        for gradient in loss_gradients[samples_idx]:
            plot_loss_gradients.append(np.max(np.abs(gradient)))
            # plot_loss_gradients.append(np.linalg.norm(gradient))
            plot_samples.append(n_samples)
        print(plot_loss_gradients[-100:])
    df = pd.DataFrame(data={"loss_gradients": plot_loss_gradients,"n_samples": plot_samples})
    print(df.head())

    matplotlib.rc('font', **{'weight': 'bold', 'size': 20})
    plot = plt.figure(num=None, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
    im = sns.boxenplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, palette="YlGnBu_d",
                       k_depth="proportion")
    plt.ylabel(r"Expected Gradients $l_\infty$-norm ($|\nabla L(x,w_i)|_\infty$)")
    plt.xlabel("Samples involved in expectations ($w_i \sim p(w|D)$)")
    # plt.ylim(0,np.max(plot_loss_gradients))

    filename = "expLossGradients_inputs=" + str(n_inputs) + "_boxenplot_"+str(dataset_name)+"_"+str(model_idx)+".png"
    os.makedirs(os.path.dirname(RESULTS+"plots/"), exist_ok=True)
    plot.savefig(RESULTS +"plots/"+ filename, dpi=150)


def final_plot(n_inputs, n_samples_list, relpath):
    matplotlib.rc('font', **{'weight': 'bold', 'size': 10})
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=120, facecolor='w', edgecolor='k')

    # mnist
    loss_gradients = []
    model_idx, dataset = 2, "mnist"
    for samples in n_samples_list:
        loss_gradients.append(load_loss_gradients(dataset_name=dataset, n_inputs=n_inputs, n_samples=samples,
                                                            model_idx=model_idx, relpath=DATA_PATH))
    plot_loss_gradients = []
    plot_samples = []
    for samples_idx, n_samples in enumerate(n_samples_list):
        print("\n\nsamples = ", n_samples, end="\t")
        for gradient in loss_gradients[samples_idx]:
            plot_loss_gradients.append(np.max(np.abs(gradient)))
            plot_samples.append(n_samples)
        print(plot_loss_gradients[-100:])
    df = pd.DataFrame(data={"loss_gradients": plot_loss_gradients, "n_samples": plot_samples})
    print(df.head())
    sns.boxenplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, palette="YlGnBu_d",
                       k_depth="proportion", ax=ax1)
    ax1.set_ylabel(r"Expected Gradients $l_\infty$-norm ($|\nabla L(x,w_i)|_\infty$)")
    ax1.set_xlabel("Samples involved in expectations ($w_i \sim p(w|D)$)")

    # fashion mnist
    loss_gradients = []
    model_idx, dataset = 5, "fashion_mnist"
    for samples in n_samples_list:
        loss_gradients.append(load_loss_gradients(dataset_name=dataset, n_inputs=n_inputs, n_samples=samples,
                                                      model_idx=model_idx, relpath=relpath))
    plot_loss_gradients = []
    plot_samples = []
    for samples_idx, n_samples in enumerate(n_samples_list):
        print("\n\nsamples = ", n_samples, end="\t")
        for gradient in loss_gradients[samples_idx]:
            plot_loss_gradients.append(np.max(np.abs(gradient)))
            plot_samples.append(n_samples)
        print(plot_loss_gradients[-100:])
    df = pd.DataFrame(data={"loss_gradients": plot_loss_gradients, "n_samples": plot_samples})
    print(df.head())

    sns.boxenplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, palette="YlGnBu_d",
                       k_depth="proportion", ax=ax2)
    ax2.set_ylabel(r"Expected Gradients $l_\infty$-norm ($|\nabla L(x,w_i)|_\infty$)")
    ax2.set_xlabel("Samples involved in expectations ($w_i \sim p(w|D)$)")

    filename = "expLossGradients_inputs=" + str(n_inputs) + "_final_boxenplot.png"
    os.makedirs(os.path.dirname(RESULTS + "plots/"), exist_ok=True)
    fig.savefig(RESULTS + "plots/" + filename, dpi=150)


def main():

    # final_plot(n_inputs=1000, n_samples_list=[1,10,50,100], relpath=DATA_PATH)
    # exit()

    # n_inputs, n_samples_list, model_idx, dataset, relpath = 1000, [1,10,50,100,500], 2, "mnist", DATA_PATH
    # n_inputs, n_samples_list, model_idx, dataset, relpath = 1000, [1, 10, 50, 100,500], 5, "fashion_mnist", DATA_PATH

    n_inputs, n_samples_list, model_idx, dataset, relpath = 100, [1,10,50,100,500], 2, "mnist", RESULTS
    # n_inputs, n_samples_list, model_idx, dataset, relpath = 100, [1, 10, 50,100], 5, "fashion_mnist", RESULTS
    # n_inputs, n_samples_list, model_idx, dataset, relpath = 1000, [1,10], 5, "fashion_mnist", RESULTS

    exp_loss_gradients = []
    for samples in n_samples_list:
        # filename = str(dataset)+"_inputs="+str(n_inputs)+"_samples="+str(samples)+"_loss_grads_"+str(model_idx)+".pkl"
        # exp_loss_gradients.append(load_from_pickle(path=DATA_PATH+str(dataset)+"/"+filename))
        exp_loss_gradients.append(load_loss_gradients(dataset_name=dataset, n_inputs=n_inputs, n_samples=samples,
                                                      model_idx=model_idx, relpath=relpath))

    plot_exp_loss_gradients_norms(loss_gradients=exp_loss_gradients, n_inputs=n_inputs,
                               n_samples_list=n_samples_list,  dataset_name=dataset, model_idx=model_idx)




if __name__ == "__main__":
    main()
