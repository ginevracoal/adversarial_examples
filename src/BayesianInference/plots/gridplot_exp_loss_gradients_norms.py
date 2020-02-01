import sys
sys.path.append(".")
from directories import *

from utils import load_from_pickle, save_to_pickle
import pyro
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


def main():

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
    exit()

    # === final plots ===

    n_samples_list = [1, 10, 50, 100]
    for dataset in ["mnist", "fashion_mnist"]:

        single_image_gradients_norms = load_from_pickle(DATA_PATH+"heatmaps/"+"single_grad_"+str(dataset)+".pkl")
        fig = plot_vanishing_gradients(gradients=single_image_gradients_norms, n_samples_list=n_samples_list)
        fig.savefig(RESULTS+"plots/"+"expLossGradients_vanishingImage_"+str(dataset)+".png")

        avg_gradients_norms = load_from_pickle(DATA_PATH+"heatmaps/"+"avg_grads_"+str(dataset)+".pkl")
        fig = plot_vanishing_gradients(gradients=avg_gradients_norms, n_samples_list=n_samples_list)
        fig.savefig(RESULTS+"plots/"+"expLossGradients_avgOverImages_"+str(dataset)+".png")


if __name__ == "__main__":
    main()
