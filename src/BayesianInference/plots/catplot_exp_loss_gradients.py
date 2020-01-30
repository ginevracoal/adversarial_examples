import sys
sys.path.append(".")
from directories import *
import argparse
import pyro
import pandas as pd
import matplotlib
from BayesianInference.adversarial_attacks import *
from BayesianInference.loss_gradients import load_loss_gradients


DATA_PATH="../data/exp_loss_gradients/"

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


def catplot_exp_loss_gradients(loss_gradients, n_inputs, n_samples_list, dataset_name, model_idx):

    plot_loss_gradients = []
    plot_samples = []

    for samples_idx, n_samples in enumerate(n_samples_list):
        for gradient in loss_gradients[samples_idx]:
            plot_loss_gradients.append(np.max(np.abs(gradient))+1)
            plot_samples.append(n_samples)
    df = pd.DataFrame(data={"loss_gradients": plot_loss_gradients,"n_samples": plot_samples})
    print(df.head())

    matplotlib.rc('font', **{'weight': 'bold', 'size': 20})
    plot = plt.figure(num=None, figsize=(14, 8), dpi=120, facecolor='w', edgecolor='k')
    im = sns.boxenplot(x="n_samples", y="loss_gradients", data=df, linewidth=-0.1, palette="YlGnBu_d",
                       k_depth="trustworthy", outlier_prop=0.7)
    im.set(yscale="log")

    if dataset_name == "mnist":
        dataset_title = "MNIST"
    elif dataset_name == "fashion_mnist":
        dataset_title = "Fashion MNIST"
    else:
        dataset_title = str(dataset_name)

    # plt.title(f"Expectation of the Gradient on {n_inputs} images from {dataset_title} dataset")
    plt.ylabel(r"Expected Gradients $l_\infty$-norm ($|\nabla L(x,w_i)|_\infty$)")
    plt.xlabel("Samples involved in expectations ($w_i \sim p(w|D)$)")

    os.makedirs(os.path.dirname(RESULTS+"catplots/"), exist_ok=True)
    filename = "expLossGradients_inputs=" + str(n_inputs) + "_catplot_"+str(dataset_name)+"_"+str(model_idx)+".png"
    plot.savefig(RESULTS +"plots/"+ filename, dpi=150)



def plot_loss_robustness(loss_gradients, n_samples_list):







def main(args):

    # === OLD DATA === # todo: deprecated, remove
    # = too few samples and inputs: =
    # n_inputs, n_samples_list, models_list, eps_list, dataset = 10, [1, 10, 50, 100, 500], [0,1,2], [0.1, 0.3, 0.6], "mnist"
    # n_inputs, n_samples_list, models_list, eps_list, dataset = 100, [1, 10, 50, 100], [0,1,2], [0.1, 0.3, 0.6], "mnist"

    # = final plot for paper =
    # n_inputs, n_samples_list, models_list, eps_list, dataset = 1000, [1, 10, 50, 100], [2], [0.1], "mnist"
    #
    #
    # for model_idx in models_list:
    #     for eps in eps_list:
    #         exp_loss_gradients, n_samples_list = load_multiple_loss_gradients(dataset_name=dataset, eps=eps,
    #                                                                           n_inputs=n_inputs, model_idx=model_idx)
    #         catplot_exp_loss_gradients(loss_gradients=exp_loss_gradients, n_inputs=n_inputs,
    #                                    n_samples_list=n_samples_list, dataset_name=dataset, model_idx=model_idx)

    # === NEW DATA ===

    n_inputs, n_samples_list, model_idx, dataset = 1000, [1,5,10,50,100], 2, "mnist"
    # n_inputs, n_samples_list, model_idx, dataset = 1000, [1, 10, 50, 100], 5, "fashion_mnist"

    exp_loss_gradients = []
    for samples in n_samples_list:
        # filename = str(dataset)+"_inputs="+str(n_inputs)+"_samples="+str(samples)+"_loss_grads_"+str(model_idx)+".pkl"
        # exp_loss_gradients.append(load_from_pickle(path=DATA_PATH+str(dataset)+"/"+filename))
        exp_loss_gradients.append(load_loss_gradients(dataset_name=dataset, n_inputs=n_inputs, n_samples=samples,
                                                      model_idx=model_idx))

    catplot_exp_loss_gradients(loss_gradients=exp_loss_gradients, n_inputs=n_inputs,
                               n_samples_list=n_samples_list,  dataset_name=dataset, model_idx=model_idx)

    boxenplot_loss(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list)
    boxenplot_robustness(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--inputs", default=100, type=int)

    main(args=parser.parse_args())
