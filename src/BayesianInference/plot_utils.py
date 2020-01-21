import sys
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
    loss_gradients = load_from_pickle(path=RESULTS+"bnn/"+filename)

    plt.subplots(figsize=(10, 6), dpi=200)
    sns.set_palette("RdBu")
    for n_inputs_idx in range(len(loss_gradients)):
        # take the maximum number of samples
        sns.distplot(loss_gradients[n_inputs_idx][-1]["avg_loss_gradient"],  hist=False, rug=True,
                     kde_kws={'shade': True, 'linewidth': 2}, label=loss_gradients[n_inputs_idx][-1]["n_samples"])

    # plt.title('Density Plot with Rug Plot for Alaska Airlines')
    plt.ylabel('Density')

    plt.legend()
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS + "distPlot_avgOverImages.png")


def plot_gradients_increasing_inputs(posterior, n_samples, device, dataset_name="mnist", mode="vi"):
    random.seed(0)
    posterior = copy.deepcopy(posterior)

    loss_gradients_increasing_inputs = []
    for n_inputs in [10, 20, 30]:
        train_loader, _, _, _ = data_loaders(dataset_name=dataset_name, batch_size=128, n_inputs=n_inputs, shuffle=True)
        posterior.evaluate(data_loader=train_loader, n_samples=n_samples)
        loss_gradients = expected_loss_gradients(posterior=posterior, n_samples=n_samples, data_loader=train_loader,
                                                device=device, mode=mode)
        last_gradients = loss_gradients[len(loss_gradients)-8:len(loss_gradients)]
        reshaped_gradients = [{"image":image.reshape([28,28,1])} for image in last_gradients]
        loss_gradients_increasing_inputs.append(reshaped_gradients)
    images = np.array(loss_gradients_increasing_inputs)
    plot_images_grid(images, path=RESULTS, filename="mnist_lossGradients_increasingInputs.png")


def plot_avg_over_images_grid(filename="avgOverImages.pkl"):
    images = load_from_pickle(path=RESULTS+"bnn/"+filename)
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

def plot_gradients_on_single_image(data_loader, posterior, n_samples_list, device, dataset_name="mnist"):
    random.seed(0)
    # pyro.clear_param_store()
    posterior = copy.deepcopy(posterior)
    accuracy_over_samples = [posterior.evaluate(data_loader=data_loader, n_samples=n_samples)
                             for n_samples in n_samples_list]

    gradients_over_images = []
    for images, labels in data_loader:
        for i in range(len(images)-5,len(images)):
            image = images[i]
            label = labels[i]
            gradients_over_samples = []
            for idx, n_samples in enumerate(n_samples_list):
                loss_gradient = expected_loss_gradient(posterior=posterior, n_samples=n_samples, image=image,
                                                       label=label, device=device, mode="vi").cpu().detach()
                gradients_over_samples.append({"image":np.array(loss_gradient).reshape([28,28,1]),"n_samples":n_samples,
                                               "lab":label,"accuracy":accuracy_over_samples[idx]})
                del loss_gradient
            gradients_over_images.append(gradients_over_samples)

    loss_gradients = np.array(gradients_over_images)
    filename = "loss_gradients_singleImage_increasingSamples_"+str(dataset_name)+".png"
    plot_images_grid(loss_gradients, path=RESULTS, filename=filename)

def plot_exp_loss_gradients_norms(exp_loss_gradients, n_inputs, n_samples_list, pnorm=2):
    exp_loss_gradients_norms = []
    for idx, n_samples in enumerate(n_samples_list):
        # print(exp_loss_gradients[idx].shape)
        # exit()
        norms = [np.linalg.norm(x=gradient, ord=pnorm).item() for gradient in exp_loss_gradients[idx]]
        exp_loss_gradients_norms.append(norms)

    print("exp_loss_gradients_norms.shape =", np.array(exp_loss_gradients_norms).shape)
    yticks = [n_samples for n_samples in n_samples_list]
    plot_heatmap(columns=exp_loss_gradients_norms, path=RESULTS,
                 filename="exp_loss_gradients_norms_inputs=" + str(n_inputs) + "_pnorm=" + str(pnorm) + "_heatmap.png",
                 xlab="image idx", ylab="n. posterior samples", yticks=yticks,
                 title="Expected loss gradients norms on {} images ".format(n_inputs))


def plot_partial_derivatives(dataset_name, n_inputs, n_samples_list, n_posteriors=1, rel_path=RESULTS):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    exp_loss_grad = []
    n_samples_column = []
    for i, n_samples in enumerate(n_samples_list):
        filename = "expLossGradients_samples=" + str(n_samples) + "_inputs=" + str(n_inputs) \
                   + "_posteriors=" + str(n_posteriors) + ".pkl"
        loss_gradients = load_from_pickle(path=rel_path + "bnn/" + filename)
        avg_partial_derivatives = loss_gradients.mean(0).log().cpu().detach().numpy()  # .log()
        # for j in range(loss_gradients.shape[0]): # images
        for k in range(len(avg_partial_derivatives)):  # partial derivatives
            exp_loss_grad.append(avg_partial_derivatives[k])
            n_samples_column.append(n_samples * n_posteriors)

    df = pd.DataFrame(data={"log(loss partial derivatives)": exp_loss_grad, "n_samples": n_samples_column})
    print(df.head())
    # print(df.describe(include='all'))

    filename = "partial_derivatives_inputs=" + str(n_inputs) \
               + "_posteriors=" + str(n_posteriors) + "_catplot.png"

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
            # axs[-1,col].set_xlabel("samples={}".format(images[-1][col]["n_samples"]), fontsize=16)

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
