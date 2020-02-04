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
    # plot for loss gradients norms

    matplotlib.rc('font', **{'weight': 'bold', 'size': 11})
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=150, facecolor='w', edgecolor='k')

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
            plot_loss_gradients.append(np.max(np.abs(gradient))+1)
            plot_samples.append(n_samples)
        print(plot_loss_gradients[-100:])
    df = pd.DataFrame(data={"loss_gradients": plot_loss_gradients, "n_samples": plot_samples})
    print(df.head())
    # sns.boxenplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, palette="YlGnBu_d",
    #                    k_depth="proportion", ax=ax1,outlier_prop=0.0, dodge=False)
    sns.violinplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, palette="YlGnBu_d", ax=ax1)
    # sns.boxplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, palette="YlGnBu_d", ax=ax1)
    ax1.set_ylabel(r"Expected Gradients $l_\infty$-norm ($|\nabla L(x,w_i)|_\infty$)")
    ax1.set_xlabel("Samples involved in expectations ($w_i \sim p(w|D)$)")
    ax1.set_yscale('log')

    # fashion mnist
    loss_gradients = []
    model_idx, dataset = 5, "fashion_mnist"
    for samples in n_samples_list:
        loss_gradients.append(load_loss_gradients(dataset_name=dataset, n_inputs=n_inputs, n_samples=samples,
                                                      model_idx=model_idx, relpath=relpath))
    loss_gradients_components = []
    plot_samples = []
    for samples_idx, n_samples in enumerate(n_samples_list):
        print("\n\nsamples = ", n_samples, end="\t")
        for gradient in loss_gradients[samples_idx]:
            loss_gradients_components.append(np.max(np.abs(gradient))+1)
            plot_samples.append(n_samples)
        print(plot_loss_gradients[-100:])
    df = pd.DataFrame(data={"loss_gradients": plot_loss_gradients, "n_samples": plot_samples})
    print(df.head())

    sns.boxenplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, palette="YlGnBu_d",
                       k_depth="proportion", ax=ax2)
    # sns.violinplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, palette="YlGnBu_d", ax=ax2)

    ax2.set_ylabel(r"Expected Gradients $l_\infty$-norm ($|\nabla L(x,w_i)|_\infty$)")
    ax2.set_xlabel("Samples involved in expectations ($w_i \sim p(w|D)$)")
    ax2.set_yscale('log')

    filename = "expLossGradients_inputs=" + str(n_inputs) + "_boxenplot.png"
    os.makedirs(os.path.dirname(RESULTS + "plots/"), exist_ok=True)
    fig.savefig(RESULTS + "plots/" + filename)


def final_plot_gradient_components(n_inputs, n_samples_list, relpath):

    matplotlib.rc('font', **{'weight': 'bold', 'size': 12})
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=150, facecolor='w', edgecolor='k')

    for col_idx, (model_idx, dataset) in enumerate([(2,"mnist"),(5,"fashion_mnist")]):
        loss_gradients = []
        for samples in n_samples_list:
            loss_gradients.append(load_loss_gradients(dataset_name=dataset, n_inputs=n_inputs, n_samples=samples,
                                                      model_idx=model_idx, relpath=relpath))
        loss_gradients_components = []
        plot_samples = []
        for samples_idx, n_samples in enumerate(n_samples_list):
            print("\n\nsamples = ", n_samples, end="\t")
            # avg_loss_gradient = loss_gradients[samples_idx].mean(0)
            avg_loss_gradient = np.array(loss_gradients[samples_idx]+1).flatten()#.mean(0)
            loss_gradients_components.extend(avg_loss_gradient)
            plot_samples.extend(np.repeat(n_samples, len(avg_loss_gradient)))
            # print(loss_gradients_components[-100:])
            print(len(loss_gradients),len(loss_gradients[0]),loss_gradients[0].shape, len(loss_gradients_components))
        # print(len(loss_gradients_components),len(loss_gradients_components[0]),loss_gradients_components)

        df = pd.DataFrame(data={"loss_gradients": loss_gradients_components, "n_samples": plot_samples})
        print(df.head())
        # sns.boxenplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, palette="YlOrRd",
        #                    k_depth="proportion", ax=ax[col_idx],outlier_prop=0.0, dodge=False)
        sns.stripplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, palette="YlOrRd", ax=ax[col_idx])

        ax[col_idx].set_ylabel("")
        ax[col_idx].set_xlabel("")
        # ax[col_idx].set_yscale('log')

    ax[0].set_title("MNIST", fontsize=10)
    ax[1].set_title("Fashion MNIST", fontsize=10)

    fig.text(0.5, 0.01, "Samples involved in the expectations ($w \sim p(w|D)$)", ha='center')
    fig.text(0.03, 0.5, r"Expected Gradients components $\langle\nabla L(x,w)\rangle_{w}$", va='center', rotation='vertical')

    filename = "expLossGradients_inputs=" + str(n_inputs) + "_boxenplot.png"
    os.makedirs(os.path.dirname(RESULTS + "plots/"), exist_ok=True)
    fig.savefig(RESULTS + "plots/" + filename)

def main():

    # final_plot(n_inputs=1000, n_samples_list=[1,10,50,100,500], relpath=DATA_PATH)
    final_plot_gradient_components(n_inputs=1000, n_samples_list=[1,10,50,100,500], relpath=DATA_PATH)
    exit()

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
