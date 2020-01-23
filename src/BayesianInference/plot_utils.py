import sys

from BayesianInference.pyro_utils import slice_data_loader

sys.path.append(".")
from directories import *
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import torch
from utils import load_from_pickle, plot_heatmap
import random
import copy
from BayesianInference.loss_gradients import expected_loss_gradients, expected_loss_gradient
import matplotlib.colors as mc


def distplot_avg_gradients_over_inputs(filename):
    loss_gradients = load_from_pickle(path=RESULTS+"bnn/"+filename+".pkl")

    plt.subplots(figsize=(10, 6), dpi=200)
    sns.set_palette("RdBu")
    for n_samples_idx in range(len(loss_gradients[0])):
        gradients = loss_gradients[-1][n_samples_idx]
    # for n_inputs_idx in range(len(loss_gradients)):
    #     gradients = loss_gradients[n_inputs_idx][-1]
    #     print(gradients["avg_loss_gradient"])
        ax = sns.distplot(gradients["avg_loss_gradient"],  hist=False, rug=True,
                     kde_kws={'shade': True, 'linewidth': 2}, kde=True,
                     label=gradients["n_samples"])
        ax.set_yscale('log')

    plt.ylabel('log(Density)')

    plt.legend()
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS + "distPlot_avgOverImages.png")


def plot_gradients_increasing_inputs(posterior, n_samples_list, n_inputs_list, device, data_loader, mode="vi"):

    for n_samples in n_samples_list:
        random.seed(0)
        posterior = copy.deepcopy(posterior)

        loss_gradients_increasing_inputs = []
        for n_inputs in n_inputs_list:
            data_loader_slice = slice_data_loader(data_loader, slice_size=n_inputs)
            # posterior.evaluate(data_loader=data_loader_slice, n_samples=n_samples)
            loss_gradients = expected_loss_gradients(posterior=posterior, n_samples=n_samples,
                                                     data_loader=data_loader_slice, device=device, mode=mode)
            last_gradients = loss_gradients[len(loss_gradients)-8:len(loss_gradients)]
            reshaped_gradients = [{"image":image.reshape([28,28,1])} for image in last_gradients]
            loss_gradients_increasing_inputs.append(reshaped_gradients)
        images = np.array(loss_gradients_increasing_inputs)
        plot_images_grid(images, path=RESULTS, filename="mnist_lossGradients_inputs="+str(n_inputs_list)+".png")


def plot_avg_over_images_grid(filename):
    images = load_from_pickle(path=RESULTS+"bnn/"+filename+".pkl")
    # print(images.shape)
    fig, axs = plt.subplots(nrows=images.shape[0], ncols=images.shape[1], figsize=(15,10))

    for col in range(images.shape[1]):
        for row in range(images.shape[0]):
            im = axs[row, col].imshow(np.squeeze(images[row][col]["avg_loss_gradient"].reshape(28,28,1)),
                                      norm=mc.Normalize(vmin=0), cmap="gray")
            axs[row, col].set_title("[{:.2f},{:.2f}]".format(np.min(images[row][col]["avg_loss_gradient"]),
                                                             np.max(images[row][col]["avg_loss_gradient"])))
            axs[row, col].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')
            axs[-1,col].set_xlabel("n_samples={}".format(images[-1][col]["n_samples"]), fontsize=14)
            axs[row,col].set_ylabel("n_inputs={}, acc={:.2f}".format(images[row][col]["n_inputs"],
                                                                 images[row][col]["accuracy"]), fontsize=14)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    fig.savefig(RESULTS + "expectedLossGradients_avgOverImages.png")


def plot_expectation_over_images(dataset_name, n_inputs, n_samples_list, rel_path=RESULTS):
    avg_loss_gradients = []
    for n_samples in n_samples_list:
        filename = "expLossGradients_samples=" + str(n_samples) + "_inputs=" + str(n_inputs)
        expected_loss_gradients = load_from_pickle(path=rel_path + "bnn/" + filename + ".pkl")
        avg_loss_gradient = np.mean(expected_loss_gradients, axis=0) / n_inputs
        avg_loss_gradients.append(avg_loss_gradient)
        print("\nn_samples={} \navg_loss_gradient[:10]={}".format(n_samples, avg_loss_gradient[:10]))

    filename = "hidden_vi_" + str(dataset_name) + "_inputs=" + str(n_inputs)
    plot_heatmap(columns=avg_loss_gradients, path=RESULTS, filename=filename + "_heatmap.png",
                 xlab="pixel idx", ylab="n. posterior samples", yticks=n_samples_list,
                 title="Expected loss gradients over {} images".format(n_inputs))

def plot_gradients_on_selected_images(loss_gradients, max_n_images, n_samples_list, filename, dataset_name="mnist"):

    # images_idxs = random.sample(range(0, len(loss_gradients[0])), n_images)
    n_images = min(len(loss_gradients[0]), max_n_images)
    gradients_along_samples = []
    for idx, n_samples in enumerate(n_samples_list):
        loss_gradients_fixed_samples = loss_gradients[idx]
        gradients_along_images = []
        for idx in range(n_images):
            selected_gradient = loss_gradients_fixed_samples[idx]
            gradients_along_images.append({"image":np.array(selected_gradient).reshape([28,28,1]),
                                           "n_samples":n_samples})
        gradients_along_samples.append(gradients_along_images)

    loss_gradients = np.array(gradients_along_samples)
    filename = filename+"expLossGradients_onImages_increasingSamples_"+str(dataset_name)+".png"
    plot_images_grid(loss_gradients, path=RESULTS, filename=filename)

def plot_gradients_on_images(loss_gradients, max_n_images, n_samples_list, filename, dataset_name="mnist"):
    loss_gradients = np.transpose(loss_gradients, axes=(1,0,2))

    vanishing_gradients_idxs = []
    for image_idx, image_gradients in enumerate(loss_gradients):
        gradient_norm = np.linalg.norm(image_gradients[0])
        if gradient_norm != 0.0:
            count_samples_idx = 0
            for samples_idx, n_samples in enumerate(n_samples_list):
                new_gradient_norm = np.linalg.norm(image_gradients[samples_idx])
                if new_gradient_norm <= gradient_norm:
                    print(new_gradient_norm)
                    gradient_norm = copy.deepcopy(new_gradient_norm)
                    count_samples_idx += 1
            if count_samples_idx == len(n_samples_list):
                vanishing_gradients_idxs.append(image_idx)
                print("\n")

    print("vanishing_gradients_idxs =", vanishing_gradients_idxs)
    vanishing_gradients = np.transpose(loss_gradients[vanishing_gradients_idxs], axes=(1,0,2))
    plot_gradients_on_selected_images(loss_gradients=vanishing_gradients, max_n_images=max_n_images,
                                    n_samples_list=n_samples_list, dataset_name="mnist",
                                      filename=filename+"_vanishing")

    non_vanishing_gradients_idxs = []
    for image_idx, image_gradients in enumerate(loss_gradients):
        gradient_norm = np.linalg.norm(image_gradients[0])
        if gradient_norm != 0.0:
            count_samples_idx = 0
            for samples_idx, n_samples in enumerate(n_samples_list):
                new_gradient_norm = np.linalg.norm(image_gradients[samples_idx])
                if new_gradient_norm >= gradient_norm:
                    print(new_gradient_norm)
                    gradient_norm = copy.deepcopy(new_gradient_norm)
                    count_samples_idx += 1
            if count_samples_idx == len(n_samples_list):
                non_vanishing_gradients_idxs.append(image_idx)
                print("\n")

    print("non_vanishing_gradients_idxs =", non_vanishing_gradients_idxs)
    non_vanishing_gradients = np.transpose(loss_gradients[non_vanishing_gradients_idxs], axes=(1,0,2))
    plot_gradients_on_selected_images(loss_gradients=non_vanishing_gradients, max_n_images=max_n_images,
                                      n_samples_list=n_samples_list, dataset_name="mnist",
                                      filename=filename+"_nonVanishing")
    # gradients_along_samples = []
    # for idx, n_samples in enumerate(n_samples_list):
    #     loss_gradients_fixed_samples = loss_gradients[idx]
    #     gradients_along_images = []
    #     for idx in images_idxs:
    #         selected_gradient = loss_gradients_fixed_samples[idx]
    #         gradients_along_images.append({"image":np.array(selected_gradient).reshape([28,28,1]),
    #                                        "n_samples":n_samples})
    #     gradients_along_samples.append(gradients_along_images)
    #
    # loss_gradients = np.array(gradients_along_samples)
    # filename = "expLossGradients_onImages_increasingSamples_"+str(dataset_name)+".png"
    # plot_images_grid(loss_gradients, path=RESULTS, filename=filename)

def plot_exp_loss_gradients_norms(exp_loss_gradients, n_inputs, n_samples_list, model_idx, filename, pnorm=2):
    exp_loss_gradients_norms = []
    for idx, n_samples in enumerate(n_samples_list):
        # print(exp_loss_gradients[idx].shape)
        # exit()
        norms = [np.linalg.norm(x=gradient, ord=pnorm).item() for gradient in exp_loss_gradients[idx]]
        exp_loss_gradients_norms.append(norms)

    print("\nexp_loss_gradients_norms.shape =", np.array(exp_loss_gradients_norms).shape)
    yticks = [n_samples for n_samples in n_samples_list]

    plot_heatmap(columns=exp_loss_gradients_norms, path=RESULTS,
                 filename=filename+"_pnorm=" + str(pnorm) + "_norms_model="+str(model_idx)+".png",
                 xlab="image idx", ylab="n. posterior samples", yticks=yticks,
                 title="Expected loss gradients norms on {} images ".format(n_inputs))

    def distplot(gradients_norms_samples, n_samples_list):
        n_inputs = len(gradients_norms_samples[0])
        plt.subplots(figsize=(10, 6), dpi=200)
        sns.set_palette("RdBu")
        for idx, n_samples in enumerate(n_samples_list):
            gradients_norms = gradients_norms_samples[idx]
            ax = sns.distplot(gradients_norms, hist=False, rug=True,
                              kde_kws={'shade': True, 'linewidth': 2},# kde=True,
                              label=str(n_samples))
            # ax.set_xscale('log')
            plt.ylabel('Density')
            plt.xlabel('norm')
            plt.title(f"Expected loss gradients norms on {n_inputs} inputs")

            plt.legend(title="n_samples")
            os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
            plt.savefig(RESULTS+"expLossGradients_norms_distplot_inputs="+str(n_inputs)+"model="+str(model_idx)+".png")

    distplot(exp_loss_gradients_norms, n_samples_list)

def catplot_pointwise_softmax_differences(dataframe, filename, epsilon_list):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(ncols=3, nrows=1)

    for i, eps in enumerate(epsilon_list):
        data = dataframe.loc[dataframe['epsilon'] == eps]
        # print(data.head())
        sns.catplot(data=data, y="softmax_difference_norms", x="n_samples", kind="boxen", ax=axs[i])

    # plot.fig.set_figheight(6)
    # plot.fig.set_figwidth(10)

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS + filename+".png", dpi=200)

def catplot_partial_derivatives(filename, n_inputs, n_samples_list, rel_path=RESULTS):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    exp_loss_grad = []
    n_samples_column = []
    for i, n_samples in enumerate(n_samples_list):
        loss_gradients = load_from_pickle(path=rel_path + "bnn/" + filename+".pkl")
        loss_gradients = torch.tensor(loss_gradients)
        avg_partial_derivatives = loss_gradients.mean(0).log()#.cpu().detach().numpy()#log()
        # for j in range(loss_gradients.shape[0]): # images
        for k in range(len(avg_partial_derivatives)):  # partial derivatives
            exp_loss_grad.append(avg_partial_derivatives[k])
            n_samples_column.append(n_samples)

    df = pd.DataFrame(data={"log(loss partial derivatives)": exp_loss_grad, "n_samples": n_samples_column})
    print(df.head())
    # print(df.describe(include='all'))

    filename = "partial_derivatives_inputs=" + str(n_inputs) + "_catplot.png"

    plot = sns.catplot(data=df, y="log(loss partial derivatives)", x="n_samples", kind="boxen")
    plot.fig.set_figheight(8)
    plot.fig.set_figwidth(15)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plot.savefig(RESULTS + filename, dpi=100)


def plot_images_grid(images, path, filename, cmap=None, row_labels=None, col_labels=None, figsize=(15,10)):
    """
    Plots the first `n_images` images of each element in image_data_list, on different rows.

    :param image_data_list: list of arrays of images to plot
    :param cmap: colormap  = gray or None
    :param test: if True it does not hang on the image
    """
    import matplotlib.colors as mc
    fig, axs = plt.subplots(nrows=images.shape[0], ncols=images.shape[1], figsize=figsize)


    for col in range(images.shape[1]):
        for row in range(images.shape[0]):
            # print("n_samples =",images[row][col]["n_samples"],
            #       "\tmin=",np.min(images[row][col]["image"]),
            #       "\tmax=", np.max(images[row][col]["image"]))
            im = axs[row, col].imshow(np.squeeze(images[row][col]["image"]), #vmin=0, vmax=1,
                                      norm=mc.Normalize(vmin=0), cmap="gray")
            axs[row,col].set_title("[{:.2f},{:.2f}]"
                                   .format(np.min(images[row][col]["image"]),
                                           np.max(images[row][col]["image"])))
            axs[row,col].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')
            # axs[-1,col].set_xlabel("samples={}, acc={}".format(images[-1][col]["n_samples"],
            #                                                    images[-1][col]["accuracy"]), fontsize=14)
            axs[row,0].set_ylabel("samples={}".format(images[row][-1]["n_samples"]), fontsize=14)

    # images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])
    # for idx, ax in enumerate(axs.flat):
    #     im = ax.imshow(np.squeeze(images[idx]),norm=mc.Normalize(vmin=0),cmap="gray")
    #     # ax.set_title()
    #     fig.colorbar(ax)
    #
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # [x0, y0, width, height]
    # fig.colorbar(im, cax=cbar_ax)
    # cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)
    # plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + filename)

    # old
    # for row in range(len(image_data_list)):
    #     for col in range(n_images):
    #         axs[row, col].imshow(np.squeeze(image_data_list[row][col]), cmap=cmap)
    #         if labels:
    #             axs[row, 1].set_title(labels[row])


    # if test is False:
    #     # If not in testing mode, block imshow.
    #     plt.show(block=False)
    #     input("Press ENTER to exit")
    #     exit()
