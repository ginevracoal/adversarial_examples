import sys
sys.path.append(".")
from directories import *

from BayesianInference.pyro_utils import data_loaders
from utils import load_from_pickle, save_to_pickle, plot_heatmap
import pyro
import matplotlib
import os
import numpy as np
import copy
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from BayesianInference.loss_gradients import load_loss_gradients, compute_vanishing_grads_idxs


DATA_PATH="../data/exp_loss_gradients/"


def plot_vanishing_gradients(gradients, n_samples_list):

    fig, axs = plt.subplots(nrows=1, ncols=len(n_samples_list), figsize=(12, 4))
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    # axs = axs.flat

    fig.tight_layout(h_pad=2, w_pad=2)

    vmin, vmax = (np.min(gradients), np.max(gradients))

    for col_idx, samples in enumerate(n_samples_list):
        loss_gradient = gradients[col_idx].reshape(28, 28)
        sns.heatmap(loss_gradient, cmap="YlGnBu", ax=axs[col_idx], square=True, vmin=vmin, vmax=vmax,
                    cbar_kws={'shrink': 0.5})
        axs[col_idx].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')

        # norm = np.linalg.norm(x=loss_gradient, ord=2)
        # expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_2$"
        norm = np.max(np.abs(loss_gradient))
        expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_\infty$"

        axs[col_idx].set_title(f"{expr} = {norm:.3f}", fontsize=11)
        axs[col_idx].set_xlabel(f"Samples = {samples}", fontsize=10)

    return fig

def plot_single_images_vanishing_gradients(loss_gradients, n_samples_list, fig_idx):

    transposed_gradients = np.transpose(np.array(loss_gradients), axes=(1, 0, 2))
    if transposed_gradients.shape[1] != len(n_samples_list):
        raise ValueError("Second dimension should contain the number of samples.")

    # save_to_pickle(data=transposed_gradients[533], relative_path=RESULTS+"plots/",
    #                filename="single_grad_"+str(fig_idx)+".pkl")
    # save_to_pickle(data=transposed_gradients[794], relative_path=RESULTS+"plots/",
    #                filename="single_grad_"+str(fig_idx)+".pkl")

    vanishing_idxs = compute_vanishing_grads_idxs(transposed_gradients, n_samples_list=n_samples_list)
    vanishing_loss_gradients = transposed_gradients[vanishing_idxs]

    for im_idx, im_gradients in enumerate(vanishing_loss_gradients):

        fig = plot_vanishing_gradients(im_gradients, n_samples_list)
        dir=RESULTS+"plots/vanishing_gradients/"
        os.makedirs(os.path.dirname(dir), exist_ok=True)
        fig.savefig(dir+"expLossGradients_vanishingImage_"+str(fig_idx)+"_"+str(im_idx)+".png")


def plot_avg_gradients_grid(loss_gradients, n_samples_list, fig_idx):

    avg_loss_gradients_over_images = np.mean(np.array(loss_gradients), axis=1)

    save_to_pickle(data=avg_loss_gradients_over_images, relative_path=RESULTS+"plots/",
                   filename="avg_grads_"+str(fig_idx)+".pkl")

    fig = plot_vanishing_gradients(avg_loss_gradients_over_images, n_samples_list)
    dir = RESULTS+"plots/"
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    fig.savefig(dir + "expLossGradients_avgOverImages_"+str(fig_idx)+".png")


def big_plot(n_samples_list):

    fig, axs = plt.subplots(nrows=4, ncols=len(n_samples_list), figsize=(10, 8))

    rows = []
    for dataset in ["mnist", "fashion_mnist"]:
        single_image_gradients_norms = load_from_pickle(DATA_PATH+"heatmaps/"+"single_grad_"+str(dataset)+".pkl")
        avg_gradients_norms = load_from_pickle(DATA_PATH+"heatmaps/"+"avg_grads_"+str(dataset)+".pkl")
        rows.append(single_image_gradients_norms)
        rows.append(avg_gradients_norms)

    for row_idx, gradients in enumerate(rows):
        for col_idx, samples in enumerate(n_samples_list):
            vmin, vmax = (np.min(gradients), np.max(gradients))
            sns.heatmap(gradients[col_idx].reshape(28, 28), ax=axs[row_idx,col_idx], cbar=col_idx == 3,
                        cmap="YlGnBu", square=True, vmin=vmin, vmax=vmax)# cbar_kws={'shrink': 1.5})
            axs[row_idx,col_idx].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')

            # norm = np.linalg.norm(x=loss_gradient, ord=2)
            # expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_2$"
            norm = np.max(np.abs(gradients[col_idx]))
            expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_\infty$"

            axs[row_idx,col_idx].set_title(f"{expr} = {norm:.3f}", fontsize=10)
            axs[3,col_idx].set_xlabel(f"Samples = {samples}", fontsize=10)

    labelpad=10
    axs[0,0].set_ylabel("Single MNIST gradient", labelpad=labelpad)
    axs[1,0].set_ylabel("Avg MNIST gradients", labelpad=labelpad)
    axs[2,0].set_ylabel("Single F. MNIST gradient", labelpad=labelpad)
    axs[3,0].set_ylabel("Avg F. MNIST gradients", labelpad=labelpad)

    fig.tight_layout()
    fig.savefig(RESULTS + "plots/" + "vanishing_loss_gradients_grid.png")

def save_chosen_images():
    test_loader = data_loaders(dataset_name="mnist", batch_size=128, n_inputs=1000, shuffle=False)[1]

    count=0
    for images, labels in test_loader:
        for i in range(len(images)):
            if count == 533:
                save_to_pickle(data=images[i], relative_path=RESULTS+"plots/", filename="mnist_chosen_image.pkl")
                fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
                sns.heatmap(images[i].reshape(28, 28), ax=ax, cmap="gray")
                fig.savefig(RESULTS + "plots/" + "mnist_chosen_image.png")
            count+=1

    test_loader = data_loaders(dataset_name="fashion_mnist", batch_size=128, n_inputs=1000, shuffle=False)[1]
    count=0
    for images, labels in test_loader:
        for i in range(len(images)):
            if count == 794:
                save_to_pickle(data=images[i], relative_path=RESULTS + "plots/",
                               filename="fashion_mnist_chosen_image.pkl")
                fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
                sns.heatmap(images[i].reshape(28, 28), ax=ax, cmap="gray")
                fig.savefig(RESULTS + "plots/" + "fashion_mnist_chosen_image.png")
            count+=1

def final_plot(n_samples_list):

    matplotlib.rc('font', **{'weight': 'bold', 'size': 11})

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 4))

    for row_idx, dataset in enumerate(["mnist", "fashion_mnist"]):
        chosen_image = load_from_pickle(DATA_PATH+"heatmaps/"+str(dataset)+"_chosen_image.pkl").reshape(28,28)
        sns.heatmap(chosen_image, ax=axs[row_idx, 0], cbar=0 == 3,
                    cmap="gray", square=True)
        axs[row_idx, 0].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')


    rows = []
    for dataset in ["mnist", "fashion_mnist"]:
        single_image_gradients_norms = load_from_pickle(DATA_PATH+"heatmaps/"+"single_grad_"+str(dataset)+".pkl")
        single_image_gradients_norms = np.delete(np.array(single_image_gradients_norms),2, axis=0)
        rows.append(single_image_gradients_norms)

    for row_idx, gradients in enumerate(rows):
        for samples_idx, samples in enumerate([1,10,100]):
            col_idx = samples_idx+1
            vmin, vmax = (np.min(gradients), np.max(gradients))
            # cmap = sns.cubehelix_palette(n_colors=10, start=0.5,rot=0.3, light=0.7, as_cmap=True)
            cmap = sns.cubehelix_palette(n_colors=10, start=0.8,rot=0.1, light=0.9, hue=1.5, as_cmap=True)
            # cmap = sns.cubehelix_palette(n_colors=10, start=0.9,rot=0.1, light=0.9, hue=1.6, as_cmap=True)
            sns.heatmap(gradients[samples_idx].reshape(28, 28), ax=axs[row_idx,col_idx], cbar=col_idx == 3,
                        cmap=cmap,
                        square=True, vmin=vmin, vmax=vmax)
            axs[row_idx,col_idx].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')

            # norm = np.linalg.norm(x=loss_gradient, ord=2)
            # expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_2$"
            norm = np.max(np.abs(gradients[samples_idx]))
            expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_\infty$"

            axs[row_idx,col_idx].set_title(f"{expr} = {norm:.3f}", fontsize=10)
            axs[1,col_idx].set_xlabel(f"Samples = {samples}", fontsize=10)

    labelpad=10
    axs[0,0].set_ylabel("MNIST", labelpad=labelpad)
    axs[1,0].set_ylabel("Fashion MNIST", labelpad=labelpad)

    # fig.tight_layout()
    os.makedirs(os.path.dirname(RESULTS + "plots/"), exist_ok=True)
    fig.savefig(RESULTS + "plots/" + "VI_heatmap.png")


def main():

    final_plot(n_samples_list = [1, 10, 50, 100])
    exit()

    # n_inputs, n_samples_list, model_idx, dataset, relpath = 100, [1,10,50,100,500], 2, "mnist", RESULTS
    # n_inputs, n_samples_list, model_idx, dataset, relpath = 100, [1,10,50,100], 5, "fashion_mnist", RESULTS

    n_inputs, n_samples_list, model_idx, dataset, relpath = 1000, [1,10,50,100], 2, "mnist", DATA_PATH
    # n_inputs, n_samples_list, model_idx, dataset, relpath = 1000, [1,10,50,100], 5, "fashion_mnist", DATA_PATH

    exp_loss_gradients = []
    for samples in n_samples_list:
        exp_loss_gradients.append(load_loss_gradients(dataset_name=dataset, n_inputs=n_inputs, n_samples=samples,
                                                      model_idx=model_idx, relpath=relpath))

    plot_single_images_vanishing_gradients(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list,
                                           fig_idx="_model="+str(model_idx)+"_samples="+str(n_samples_list))

    plot_avg_gradients_grid(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list,
                            fig_idx="_inputs="+str(n_inputs)+"_" + str(model_idx))


if __name__ == "__main__":
    main()
