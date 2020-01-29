import sys
sys.path.append(".")
from directories import *
import argparse
import pyro
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from BayesianInference.adversarial_attacks import *
from BayesianInference.pyro_utils import data_loaders, slice_data_loader
from BayesianInference.hidden_bnn import NN
from BayesianInference.loss_gradients import expected_loss_gradients, load_multiple_loss_gradients, load_loss_gradients
from BayesianInference.hidden_vi_bnn import VI_BNN
import matplotlib.colors as mc


# def load_data(eps, model_idx, n_samples_list, n_inputs):
#
#     pointwise_exp_loss_gradients = []
#
#     for samples in n_samples_list:
#
#         filename = "mnist_inputs="+str(n_inputs)+"_epsilon="+str(eps)+"_samples="+str(samples)\
#                    +"_model="+str(model_idx)+"_attack.pkl"
#         attack_dict = load_from_pickle("../data/exp_loss_gradients/"+filename)
#         exp_loss_gradients = np.array([np.array(gradient) for gradient in attack_dict["loss_gradients"]])
#         pointwise_exp_loss_gradients.append(exp_loss_gradients)
#         pointwise_exp_loss_gradients = np.array(pointwise_exp_loss_gradients)
#
#         print("\npointwise_exp_loss_gradients.shape = ",pointwise_exp_loss_gradients.shape)
#         # print("\nexp_loss_gradient.shape =", exp_loss_gradient.shape)
#
#     return pointwise_exp_loss_gradients


def catplot_exp_loss_gradients(loss_gradients, n_inputs, n_samples_list, fig_idx):
    # sns.set()
    sns.set_palette("YlGnBu", len(n_samples_list))

    plot_loss_gradients = []
    plot_samples = []

    print("\nloss_gradients.shape =", loss_gradients)
    avg_loss_gradients = np.mean(loss_gradients, axis=1) # avg over inputs
    # avg_loss_gradients = np.mean(loss_gradients, axis=2) # avg over components
    print("\navg_loss_gradients.shape =", avg_loss_gradients.shape)

    for samples_idx, n_samples in enumerate(n_samples_list):
        for gradient in avg_loss_gradients[samples_idx]:
            plot_loss_gradients.append(gradient)
            plot_samples.append(n_samples)

    df = pd.DataFrame(data={"loss_gradients": plot_loss_gradients,
                            "n_samples": plot_samples})
    print(df.head())

    filename = "expLossGradients_inputs=" + str(n_inputs) + "_catplot_"+str(fig_idx)+".png"

    plot = sns.catplot(data=df, y="loss_gradients", x="n_samples", kind="boxen")
    expr = r"$\langle\nabla_x L(x,w)\rangle_{(x,w)}$"
    plot.set_ylabels(f"Exp. loss gradients  {expr}",fontsize=12)
    plot.set_xlabels("n. posterior samples $w \sim p(w|D)$", fontsize=12)
    # plot.fig.suptitle(f"Expected loss gradients avg. over {n_inputs} inputs")

    plot.fig.set_figheight(6)
    plot.fig.set_figwidth(8)
    os.makedirs(os.path.dirname(RESULTS+"catplots/"), exist_ok=True)
    plot.savefig(RESULTS +"catplots/"+ filename, dpi=100)



def main(args):

    # === too few samples and inputs ===
    # n_inputs, n_samples_list, models_list, eps_list, dataset = 10, [1, 10, 50, 100, 500], [0,1,2], [0.1, 0.3, 0.6], "mnist"
    # n_inputs, n_samples_list, models_list, eps_list, dataset = 100, [1, 10, 50, 100], [0,1,2], [0.1, 0.3, 0.6], "mnist"

    # === final plot for paper ===
    n_inputs, n_samples_list, models_list, eps_list, dataset = 1000, [1, 10, 50, 100], [2], [0.1], "mnist"


    for model_idx in models_list:
        for eps in eps_list:
            exp_loss_gradients, n_samples_list = load_multiple_loss_gradients(dataset_name=dataset, eps=eps,
                                                                              n_inputs=n_inputs, model_idx=model_idx)
            catplot_exp_loss_gradients(loss_gradients=exp_loss_gradients, n_inputs=n_inputs,
                                       n_samples_list=n_samples_list,
                                       fig_idx="_model="+str(model_idx))


    # new tests
    # model_idx = 3
    # n_inputs, n_samples_list, models_list, eps_list, dataset = 1000, [1, 10, 50, 100], [2], [0.1], "fashion_mnist"
    # exp_loss_gradients = []
    #
    # loss_gradients = load_loss_gradients(dataset_name=dataset, eps=eps,
    #                                                                   n_inputs=n_inputs, model_idx=model_idx)
    # catplot_exp_loss_gradients(loss_gradients=exp_loss_gradients, n_inputs=n_inputs,
    #                            n_samples_list=n_samples_list,
    #                            fig_idx="_model=" + str(model_idx))

if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--inputs", default=100, type=int)

    main(args=parser.parse_args())
