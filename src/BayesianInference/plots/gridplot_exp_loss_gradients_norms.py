import sys
sys.path.append(".")
from directories import *

import argparse
import pyro
import pandas as pd
from BayesianInference.adversarial_attacks import *
from BayesianInference.pyro_utils import data_loaders, slice_data_loader
from BayesianInference.hidden_bnn import NN
from BayesianInference.loss_gradients import load_multiple_loss_gradients
from BayesianInference.hidden_vi_bnn import VI_BNN, hidden_vi_models
import matplotlib.colors as mc
import copy

# todo this should be done in the attacks script!
def create_save_attacks(dataset_name, n_inputs, n_samples_list, model_idxs, device):

    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=12, n_inputs=n_inputs, shuffle=True)

    for idx in model_idxs:
        model = hidden_vi_models[idx]
        for n_samples in n_samples_list:
            bayesnn = VI_BNN(input_shape=input_shape, device=device, architecture=model["architecture"],
                             activation=model["activation"])
            posterior = bayesnn.load_posterior(posterior_name=model["filename"],
                                               relative_path=TRAINED_MODELS,
                                               activation=model["activation"])
            filename = dataset_name + "_inputs=" + str(n_inputs) + "_epsilon=" + str(0.3) \
                       + "_samples=" + str(n_samples) + "_model=" + str(idx) + "_attack.pkl"
            bayesian_attack(model=posterior, data_loader=test_loader, epsilon=0.01, device=device,
                                          n_attack_samples=n_samples, n_pred_samples=n_samples,
                                          filename=filename)
            del bayesnn, posterior

# todo remove deprecated methods
# def create_save_gradients(dataset_name, n_inputs, n_samples_list, model_idxs, device):
#     train_loader, test_loader, data_format, input_shape = \
#         data_loaders(dataset_name=dataset_name, batch_size=12, n_inputs=n_inputs, shuffle=True)
#
#     for idx in model_idxs:
#         model = hidden_vi_models[idx]
#         for n_samples in n_samples_list:
#             bayesnn = VI_BNN(input_shape=input_shape, device=device, architecture=model["architecture"],
#                              activation=model["activation"])
#             posterior = bayesnn.load_posterior(posterior_name=model["filename"],
#                                                relative_path=TRAINED_MODELS,
#                                                activation=model["activation"])
#
#             exp_loss_gradients = expected_loss_gradients(posterior, n_samples, test_loader, device, mode="vi")
#
#             filename = dataset_name + "_inputs=" + str(n_inputs) \
#                        + "_samples=" + str(n_samples) + "_model=" + str(idx) + "_loss_gradients.pkl"
#             loss_gradients_dict = {"loss_gradients":exp_loss_gradients}
#             save_to_pickle(data=loss_gradients_dict, relative_path=RESULTS+"bnn/", filename=filename)
#
#             del bayesnn, posterior, loss_gradients_dict


# def load_slice_data(dataset_name, n_inputs, model_idx):
#
#     if n_inputs == 100:
#         eps = 0.1
#         n_samples_list = [1, 10, 50, 100]
#         data = load_data(eps=eps, model_idx=model_idx, n_samples_list=n_samples_list, n_inputs=n_inputs)
#         return data, n_samples_list
#
#     elif n_inputs == 10:
#
#         exp_loss_gradients = []
#
#         for samples in [1, 10, 50, 100]:
#             filename = str(dataset_name)+"_inputs=100_epsilon=" + str(eps) + "_samples=" + str(samples) \
#                        + "_model=" + str(model_idx) + "_attack.pkl"
#             attack_dict = load_from_pickle("../data/exp_loss_gradients/" + filename)
#             loss_gradients = np.array([np.array(gradient) for gradient in attack_dict["loss_gradients"][:n_inputs]])
#             exp_loss_gradients.append(np.squeeze(loss_gradients))
#
#         filename = str(dataset_name)+"_inputs=" + str(n_inputs) + "_epsilon=" + str(eps) + "_samples=500" \
#                    + "_model=" + str(model_idx) + "_attack.pkl"
#         attack_dict = load_from_pickle("../data/exp_loss_gradients/" + filename)
#         loss_gradients = np.array([np.array(gradient) for gradient in attack_dict["loss_gradients"]])
#         exp_loss_gradients.append(np.squeeze(loss_gradients))
#
#         exp_loss_gradients = np.array(exp_loss_gradients)
#         print("exp_loss_gradients.shape =", exp_loss_gradients.shape)
#         return exp_loss_gradients, [1, 10, 50, 100, 500]
#
#     elif n_inputs == 1000 and model_idx == 2:
#         exp_loss_gradients = []
#         # n_samples = []
#
#         model_idx = 2
#         for samples in [1, 10]:
#             filename = str(dataset_name)+"_inputs=1000_epsilon=" + str(0.1) + "_samples=" + str(samples) \
#                        + "_model=" + str(model_idx) + "_attack.pkl"
#             attack_dict = load_from_pickle("../data/exp_loss_gradients/" + filename)
#             loss_gradients = np.array([np.array(gradient) for gradient in attack_dict["loss_gradients"][:n_inputs]])
#             exp_loss_gradients.append(np.squeeze(loss_gradients))
#             # n_samples.append(samples)
#
#         for samples in [50, 100]:
#             filename = str(dataset_name) + "_inputs=" + str(n_inputs) \
#                        + "_samples=" + str(samples) + "_model=" + str(model_idx) + "_loss_gradients.pkl"
#             attack_dict = load_from_pickle("../data/exp_loss_gradients/" + filename)
#             loss_gradients = np.array([np.array(gradient) for gradient in attack_dict["loss_gradients"][:n_inputs]])
#             exp_loss_gradients.append(np.squeeze(loss_gradients))
#             # n_samples.append(samples)
#
#         exp_loss_gradients = np.array(exp_loss_gradients)
#         print("exp_loss_gradients.shape =", exp_loss_gradients.shape)
#         return exp_loss_gradients, [1,10,50,100]


# def load_data(eps, model_idx, n_samples_list, n_inputs):
#
#     exp_loss_gradients = []
#
#     for samples in n_samples_list:
#
#         filename = "mnist_inputs="+str(n_inputs)+"_epsilon="+str(eps)+"_samples="+str(samples)\
#                    +"_model="+str(model_idx)+"_attack.pkl"
#         attack_dict = load_from_pickle("../data/exp_loss_gradients/"+filename)
#         loss_gradients = np.array([np.array(gradient) for gradient in attack_dict["loss_gradients"]])
#         exp_loss_gradients.append(np.squeeze(loss_gradients))
#         print("\nloss_gradients.shape = ",loss_gradients.shape)
#
#     exp_loss_gradients = np.array(exp_loss_gradients)
#     print("exp_loss_gradients.shape =", exp_loss_gradients.shape)
#     return exp_loss_gradients


figsize = (12,4)


def plot_single_images_vanishing_gradients(loss_gradients, n_samples_list, fig_idx):
    transposed_gradients = copy.deepcopy(np.transpose(loss_gradients, axes=(1, 0, 2)))

    def compute_vanishing_grads_idxs():

        vanishing_gradients_idxs = []

        print("\nvanishing gradients norms:")
        for image_idx, image_gradients in enumerate(transposed_gradients):
            gradient_norm = np.linalg.norm(image_gradients[0])
            if gradient_norm != 0.0:
                count_samples_idx = 0
                for samples_idx, n_samples in enumerate(n_samples_list):
                    new_gradient_norm = np.linalg.norm(image_gradients[samples_idx])
                    if new_gradient_norm <= gradient_norm:
                        print(new_gradient_norm, end="\t")
                        gradient_norm = copy.deepcopy(new_gradient_norm)
                        count_samples_idx += 1
                if count_samples_idx == len(n_samples_list):
                    vanishing_gradients_idxs.append(image_idx)
                    print("\n")

        print("\nvanishing_gradients_idxs = ", vanishing_gradients_idxs)
        return vanishing_gradients_idxs

    vanishing_idxs = compute_vanishing_grads_idxs()
    selected_loss_gradients = transposed_gradients[vanishing_idxs]
    # selected_loss_gradients = transposed_gradients[26]
    print("\nselected_loss_gradients.shape =",selected_loss_gradients.shape)

    for im_idx, im_gradients in enumerate(selected_loss_gradients):

        fig, axs = plt.subplots(nrows=1, ncols=len(n_samples_list), figsize=figsize)
        # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        # axs = axs.flat
        fig.tight_layout(h_pad=2, w_pad=2)

        vmin, vmax = (np.min(im_gradients), np.max(im_gradients))

        for col_idx, samples in enumerate(n_samples_list):

            loss_gradient = im_gradients[col_idx].reshape(28, 28)
            sns.heatmap(loss_gradient, cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axs[col_idx],square=True,
                        cbar_kws={'shrink': 0.5})
            axs[col_idx].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')
            norm = np.linalg.norm(x=loss_gradient, ord=2)
            expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_2$"
            axs[col_idx].set_title(f"{expr} = {norm:.3f}", fontsize=11)
            axs[col_idx].set_xlabel(f"Samples = {samples}", fontsize=10)

        os.makedirs(os.path.dirname(RESULTS+"vanishing_gradients/"), exist_ok=True)
        fig.savefig(RESULTS+"vanishing_gradients/"+"expLossGradients_vanishingImage_"
                    +str(fig_idx)+"_"+str(im_idx)+".png")


def plot_avg_gradients_grid(loss_gradients, n_samples_list, fig_idx):

    avg_loss_gradients_over_images = np.mean(loss_gradients, axis=1)
    print("\navg_loss_gradients_over_images.shape", avg_loss_gradients_over_images.shape)

    fig, axs = plt.subplots(nrows=1, ncols=len(n_samples_list), figsize=figsize)
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    # axs = axs.flat
    fig.tight_layout(h_pad=2, w_pad=2)

    vmin=np.min(avg_loss_gradients_over_images)
    vmax=np.max(avg_loss_gradients_over_images)

    for col_idx, samples in enumerate(n_samples_list):
        loss_gradient = avg_loss_gradients_over_images[col_idx].reshape(28, 28)
        sns.heatmap(loss_gradient, cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axs[col_idx], square=True,
                    cbar_kws={'shrink': 0.5})
        avg_norm = np.linalg.norm(x=loss_gradient, ord=2)
        expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_2$"
        axs[col_idx].set_title(f"{expr} = {avg_norm:.3f}", fontsize=11)
        axs[col_idx].set_xlabel(f"Samples = {samples}", fontsize=10)
        axs[col_idx].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')

    os.makedirs(os.path.dirname(RESULTS+"vanishing_gradients/"), exist_ok=True)
    fig.savefig(RESULTS+"vanishing_gradients/" + "expLossGradients_avgOverImages_"+str(fig_idx)+".png")


def main(args):

    # n_inputs, n_samples_list, models_list, eps_list, dataset = 10, [1, 10, 50, 100, 500], [0,1,2], [0.1, 0.3, 0.6], "mnist"
    # n_inputs, n_samples_list, models_list, eps_list, dataset = 100, [1, 10, 50, 100], [0,1,2], [0.1, 0.3, 0.6], "mnist"
    n_inputs, n_samples_list, models_list, eps_list, dataset = 1000, [1, 10, 50, 100], [2], [0.1], "mnist"


    for model_idx in models_list:
        for eps in eps_list:
            exp_loss_gradients, n_samples_list = load_multiple_loss_gradients(dataset_name=dataset, eps=eps,
                                                                              n_inputs=n_inputs, model_idx=model_idx)

            # plot_single_images_vanishing_gradients(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list,
            #                                        fig_idx="_model="+str(model_idx)+"_samples="+str(n_samples_list))

            plot_avg_gradients_grid(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list,
                                    fig_idx="_eps="+str(eps)+"_"+str(model_idx))


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--inputs", default=1000, type=int)

    main(args=parser.parse_args())
