import sys
sys.path.append(".")
from directories import *

import pyro
import argparse
from BayesianInference.adversarial_attacks import *
from BayesianInference.loss_gradients import load_loss_gradients, compute_vanishing_grads_idxs


DATA_PATH="../data/exp_loss_gradients/"


def plot_single_images_vanishing_gradients(loss_gradients, n_samples_list, fig_idx):
    loss_gradients = np.array(loss_gradients)
    transposed_gradients = copy.deepcopy(np.transpose(loss_gradients, axes=(1, 0, 2)))
    vanishing_idxs = compute_vanishing_grads_idxs(transposed_gradients, n_samples_list=n_samples_list)
    vanishing_loss_gradients = transposed_gradients[vanishing_idxs]

    for im_idx, im_gradients in enumerate(vanishing_loss_gradients):

        fig, axs = plt.subplots(nrows=1, ncols=len(n_samples_list), figsize=(12,4))
        # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        # axs = axs.flat

        fig.tight_layout(h_pad=2, w_pad=2)

        vmin, vmax = (np.min(im_gradients), np.max(im_gradients))

        for col_idx, samples in enumerate(n_samples_list):

            loss_gradient = im_gradients[col_idx].reshape(28, 28)
            sns.heatmap(loss_gradient, cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axs[col_idx], square=True,
                        cbar_kws={'shrink': 0.5})
            axs[col_idx].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')

            # norm = np.linalg.norm(x=loss_gradient, ord=2)
            # expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_2$"
            norm = np.max(np.abs(loss_gradient))
            expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_\infty$"

            axs[col_idx].set_title(f"{expr} = {norm:.3f}", fontsize=11)
            axs[col_idx].set_xlabel(f"Samples = {samples}", fontsize=10)

        dir=RESULTS+"plots/vanishing_gradients/"
        os.makedirs(os.path.dirname(dir), exist_ok=True)
        fig.savefig(dir+"expLossGradients_vanishingImage_"+str(fig_idx)+"_"+str(im_idx)+".png")


def plot_avg_gradients_grid(loss_gradients, n_samples_list, fig_idx):

    avg_loss_gradients_over_images = np.mean(loss_gradients, axis=1)
    print("\navg_loss_gradients_over_images.shape", avg_loss_gradients_over_images.shape)

    fig, axs = plt.subplots(nrows=1, ncols=len(n_samples_list), figsize=(12,4))
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    # axs = axs.flat
    fig.tight_layout(h_pad=2, w_pad=2)

    vmin=np.min(avg_loss_gradients_over_images)
    vmax=np.max(avg_loss_gradients_over_images)

    for col_idx, samples in enumerate(n_samples_list):
        loss_gradient = avg_loss_gradients_over_images[col_idx].reshape(28, 28)
        sns.heatmap(loss_gradient, cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axs[col_idx], square=True,
                    cbar_kws={'shrink': 0.5})

        # norm = np.linalg.norm(x=loss_gradient, ord=2)
        # expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_2$"
        norm = np.max(np.abs(loss_gradient))
        expr = r"$|\langle\nabla_x L(x,w)\rangle_w|_\infty$"

        axs[col_idx].set_title(f"{expr} = {norm:.3f}", fontsize=11)
        axs[col_idx].set_xlabel(f"Samples = {samples}", fontsize=10)
        axs[col_idx].tick_params(left="off", bottom="off", labelleft='off', labelbottom='off')

    os.makedirs(os.path.dirname(RESULTS+"plots/"), exist_ok=True)
    fig.savefig(RESULTS+"plots/" + "expLossGradients_avgOverImages_"+str(fig_idx)+".png")


def main(args):

    # === OLD saved data ===
    # n_inputs, n_samples_list, models_list, eps_list, dataset = 10, [1, 10, 50, 100, 500], [0,1,2], [0.1, 0.3, 0.6], "mnist"
    # n_inputs, n_samples_list, models_list, eps_list, dataset = 100, [1, 10, 50, 100], [0,1,2], [0.1, 0.3, 0.6], "mnist"
    # n_inputs, n_samples_list, models_list, eps_list, dataset = 1000, [1, 10, 50, 100], [2], [0.1], "mnist"
    #
    # for model_idx in models_list:
    #     for eps in eps_list:
    #         exp_loss_gradients, n_samples_list = load_multiple_loss_gradients(dataset_name=dataset, eps=eps,
    #                                                                           n_inputs=n_inputs, model_idx=model_idx)
    #
    #         # plot_single_images_vanishing_gradients(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list,
    #         #                                        fig_idx="_model="+str(model_idx)+"_samples="+str(n_samples_list))
    #
    #         plot_avg_gradients_grid(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list,
    #                                 fig_idx="_eps="+str(eps)+"_"+str(model_idx))

    # === new data ===
    n_inputs, n_samples_list, model_idx, dataset = 1000, [1,10,100,500], 2, "mnist"
    # n_inputs, n_samples_list, model_idx, dataset = 1000, [1,10,50,100], 5, "fashion_mnist"

    exp_loss_gradients = []
    for samples in n_samples_list:
        # filename = str(dataset)+"_inputs="+str(n_inputs)+"_samples="+str(samples)+"_loss_grads_"+str(model_idx)+".pkl"
        # exp_loss_gradients.append(load_from_pickle(path=DATA_PATH+str(dataset)+"/"+filename))
        exp_loss_gradients.append(load_loss_gradients(dataset_name=dataset, n_inputs=n_inputs, n_samples=samples,
                                                      model_idx=model_idx, relpath=DATA_PATH))

    plot_single_images_vanishing_gradients(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list,
                                           fig_idx="_model="+str(model_idx)+"_samples="+str(n_samples_list))

    plot_avg_gradients_grid(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list,
                            fig_idx="_inputs="+str(n_inputs)+"_" + str(model_idx))




if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--inputs", default=1000, type=int)

    main(args=parser.parse_args())
