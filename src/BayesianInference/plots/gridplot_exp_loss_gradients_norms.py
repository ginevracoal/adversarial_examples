import sys
sys.path.append(".")
from directories import *

import argparse
import pyro
import pandas as pd
from BayesianInference.adversarial_attacks import *
from BayesianInference.pyro_utils import data_loaders, slice_data_loader
from BayesianInference.hidden_bnn import NN
from BayesianInference.loss_gradients import expected_loss_gradients
from BayesianInference.hidden_vi_bnn import VI_BNN
import matplotlib.colors as mc
import copy


# def average_over_images(posterior, n_inputs_list, n_samples_list, device, data_loader, filename, mode="vi"):
#     avg_over_images = []
#     for n_inputs in n_inputs_list:
#         data_loader_slice = slice_data_loader(data_loader=data_loader, slice_size=n_inputs)
#
#         loss_gradients = []
#         for n_samples in n_samples_list:
#             accuracy = posterior.evaluate(data_loader=data_loader_slice, n_samples=n_samples)
#             loss_gradient = expected_loss_gradients(posterior=posterior,
#                                                     n_samples=n_samples,
#                                                     data_loader=data_loader_slice,
#                                                     device=device, mode=mode)
#
#             plot_heatmap(columns=loss_gradient, path=RESULTS,
#                          filename="lossGradients_inputs="+str(n_inputs)+"_samples="+str(n_samples)+"_heatmap.png",
#                          xlab="pixel idx", ylab="image idx",
#                          title=f"Loss gradients pixel components on {n_samples} sampled posteriors")
#
#             avg_loss_gradient = np.mean(np.array(loss_gradient), axis=0)
#             loss_gradients.append({"avg_loss_gradient":avg_loss_gradient, "n_samples":n_samples, "n_inputs":n_inputs,
#                                    "accuracy":accuracy})
#         avg_over_images.append(loss_gradients)
#
#     avg_over_images = np.array(avg_over_images)
#     save_to_pickle(data=avg_over_images, relative_path=RESULTS+"bnn/", filename=filename+".pkl")

def create_save_data(dataset_name, n_inputs, n_samples_list, eps_list, model_idxs, device):

    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=12, n_inputs=n_inputs, shuffle=True)

    models_list = [{"idx":0,"filename":"hidden_vi_mnist_inputs=10000_lr=0.0002_epochs=100","activation":"softmax",
                    "dataset":"mnist"},
                   {"idx":1,"filename":"hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=100","activation":"softmax",
                    "dataset":"mnist"},
                   {"idx":2,"filename":"hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=11","activation":"softmax",
                    "dataset":"mnist"}]

    attacks = []
    for idx in model_idxs:
        for eps in eps_list:
            for n_samples in n_samples_list:
                bayesnn = VI_BNN(input_shape=input_shape, device=device, dataset_name=dataset_name,
                                 activation=models_list[idx]["activation"])
                posterior = bayesnn.load_posterior(posterior_name=models_list[idx]["filename"],
                                                   relative_path=TRAINED_MODELS,
                                                   activation=models_list[idx]["activation"])
                attack_dict = bayesian_attack(model=posterior, data_loader=test_loader, epsilon=eps, device=device,
                                                          n_attack_samples=n_samples, n_pred_samples=n_samples)
                attacks.append(attack_dict)
                filename = dataset_name+"_inputs="+str(n_inputs)+"_epsilon="+str(eps)\
                           +"_samples="+str(n_samples)+"_model="+str(idx)+"_attack.pkl"
                save_to_pickle(relative_path=RESULTS + "bnn/", filename=filename, data=attack_dict)


def load_data(eps, model_idx, n_samples_list, n_inputs):

    exp_loss_gradients = []

    # == load all data in dir ==
    # n_attack_samples = []
    # epsilon = []
    # path = "../data/exp_loss_gradients/"
    # for file in os.listdir(path):
    #     if file.endswith(".pkl"):
    #         dict = load_from_pickle(path=path+file)
    #         print(dict.keys())
    #         exp_loss_gradients.append(dict["softmax_robustness"])
    #         n_attack_samples.append(dict["n_attack_samples"])
    #         epsilon.append(dict["epsilon"])

    for samples in n_samples_list:

        filename = "mnist_inputs="+str(n_inputs)+"_epsilon="+str(eps)+"_samples="+str(samples)\
                   +"_model="+str(model_idx)+"_attack.pkl"
        attack_dict = load_from_pickle("../data/exp_loss_gradients/"+filename)
        loss_gradients = np.array([np.array(gradient) for gradient in attack_dict["loss_gradients"]])
        exp_loss_gradients.append(np.squeeze(loss_gradients))

        print("\nloss_gradients.shape = ",loss_gradients.shape)

    exp_loss_gradients = np.array(exp_loss_gradients)
    print("exp_loss_gradients.shape =", exp_loss_gradients.shape)

    return exp_loss_gradients


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
    print("\nselected_loss_gradients.shape =",selected_loss_gradients.shape)


    for im_idx, im_gradients in enumerate(selected_loss_gradients):

        fig, axs = plt.subplots(nrows=1, ncols=len(n_samples_list), figsize=(18, 6))
        # todo caption con la norma 2 del gradiente

        vmin, vmax = (np.min(im_gradients), np.max(im_gradients))

        for col_idx, samples in enumerate(n_samples_list):

            loss_gradient = im_gradients[col_idx].reshape(28, 28)
            sns.heatmap(loss_gradient, cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axs[col_idx],square=True,
                        cbar_kws={'shrink': .5})
            axs[col_idx].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')
            norm = np.linalg.norm(x=loss_gradient, ord=2)
            expr = r"$|\mathbb{E}_{x,w}[\nabla_x L(x,w)]|_2$"
            axs[col_idx].set_title(f"{expr} = {norm:.3f}", fontsize=11)
            axs[col_idx].set_xlabel(f"Samples = {samples}", fontsize=10)

        os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
        fig.savefig(RESULTS + "expLossGradients_vanishingImage_" + str(fig_idx) +"_"+str(im_idx) + ".png")

def plot_avg_gradients_grid(loss_gradients, n_samples_list, fig_idx):

    avg_loss_gradients_over_images = np.mean(loss_gradients, axis=1)
    print("\navg_loss_gradients_over_images.shape", avg_loss_gradients_over_images.shape)

    fig, axs = plt.subplots(nrows=1, ncols=len(n_samples_list), figsize=(12,6))

    vmin=np.min(avg_loss_gradients_over_images)
    vmax=np.max(avg_loss_gradients_over_images)

    for col_idx, samples in enumerate(n_samples_list):
        loss_gradient = avg_loss_gradients_over_images[col_idx].reshape(28, 28)
        sns.heatmap(loss_gradient, cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axs[col_idx], square=True,
                    cbar_kws={'shrink': .3})
        avg_norm = np.linalg.norm(x=loss_gradient, ord=2)
        expr = r"$|\mathbb{E}_{x,w}[\nabla_x L(x,w)|_2]$"
        axs[col_idx].set_title(f"{expr} = {avg_norm:.3f}", fontsize=11)
        axs[col_idx].set_xlabel(f"Samples = {samples}", fontsize=10)
        axs[col_idx].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    fig.savefig(RESULTS + "expLossGradients_avgOverImages_"+str(fig_idx)+".png")


def main(args):


    inputs, model_idxs, n_samples_list, eps_list = (10, [0,1,2], [1, 10, 50, 100], [0.1, 0.3, 0.6])

    create_save_data(dataset_name=args.dataset, n_inputs=args.inputs, n_samples_list=n_samples_list,
                     eps_list=eps_list, model_idxs=model_idxs, device=args.device)
    exit()

    # === load from the available data ===
    # inputs, model_idxs, n_samples_list, eps_list = (100, [0,1,2], [1, 10, 50, 100], [0.1, 0.3, 0.6])
    inputs, model_idxs, n_samples_list, eps_list = (10, [0,1,2], [1, 10, 50, 100, 500], [0.1, 0.3, 0.6])

    for eps in eps_list:
        for model_idx in [0,1,2]:
            fig_idx = "_eps="+str(eps)+"_"+str(model_idx)
            exp_loss_gradients = load_data(eps=eps, model_idx=model_idx, n_samples_list=n_samples_list,
                                           n_inputs=args.inputs)
            plot_single_images_vanishing_gradients(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list,
                                                   fig_idx=fig_idx)
            plot_avg_gradients_grid(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list, fig_idx=fig_idx)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--inputs", default=100, type=int)

    main(args=parser.parse_args())
