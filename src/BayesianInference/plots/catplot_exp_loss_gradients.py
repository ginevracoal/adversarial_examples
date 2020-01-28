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


def load_data(eps, model_idx, n_samples_list, n_inputs):

    pointwise_exp_loss_gradients = []

    for samples in n_samples_list:

        filename = "mnist_inputs="+str(n_inputs)+"_epsilon="+str(eps)+"_samples="+str(samples)\
                   +"_model="+str(model_idx)+"_attack.pkl"
        attack_dict = load_from_pickle("../data/exp_loss_gradients/"+filename)
        exp_loss_gradients = np.array([np.array(gradient) for gradient in attack_dict["loss_gradients"]])
        pointwise_exp_loss_gradients.append(exp_loss_gradients)
        pointwise_exp_loss_gradients = np.array(pointwise_exp_loss_gradients)

        print("\npointwise_exp_loss_gradients.shape = ",pointwise_exp_loss_gradients.shape)
        # print("\nexp_loss_gradient.shape =", exp_loss_gradient.shape)

    return pointwise_exp_loss_gradients


def catplot_exp_loss_gradients(filename, n_inputs, n_samples_list, rel_path=RESULTS):
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



def main(args):
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=32, n_inputs=args.inputs, shuffle=True)

    models_list = [{"idx":0,"filename":"hidden_vi_mnist_inputs=10000_lr=0.0002_epochs=100","activation":"softmax",
                    "dataset":"mnist"},
                   {"idx":1,"filename":"hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=100","activation":"softmax",
                    "dataset":"mnist"},
                   {"idx":2,"filename":"hidden_vi_mnist_inputs=60000_lr=0.0002_epochs=11","activation":"softmax",
                    "dataset":"mnist"}]

    # n_samples_list = [10, 500]
    # eps_list = [0.1, 0.3, 0.6]
    # attacks = []
    # count=180
    # for idx in [0,1,2]:
    #     for eps in eps_list:
    #         for n_samples in n_samples_list:
    #             bayesnn = VI_BNN(input_shape=input_shape, device=args.device, dataset_name=args.dataset,
    #                              activation=models_list[idx]["activation"])
    #             posterior = bayesnn.load_posterior(posterior_name=models_list[idx]["filename"],
    #                                                relative_path=TRAINED_MODELS,
    #                                                activation=models_list[idx]["activation"])
    #             attack_dict = bayesian_attack(model=posterior, data_loader=test_loader, epsilon=eps, device=args.device,
    #                                                       n_attack_samples=n_samples, n_pred_samples=n_samples)
    #             attacks.append(attack_dict)
    #             filename = args.dataset + "_nn_attack_" + str(count) + ".pkl"
    #             save_to_pickle(relative_path=RESULTS + "nn/", filename=filename, data=attack_dict)
    #             count += 1
    # exit()

    n_samples_list = [50, 100] # [10, 50, 100 500]
    eps_list = [0.1, 0.3, 0.6]
    for eps in eps_list:
        for model_idx in [0,1,2]:
            exp_loss_gradients = load_data(eps=eps, model_idx=model_idx, n_samples_list=n_samples_list)
            distplot(loss_gradients=exp_loss_gradients, n_samples_list=n_samples_list,
                                     fig_idx="_eps="+str(eps)+"_"+str(model_idx))
            # fig_idx += 1


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--inputs", default=100, type=int)

    main(args=parser.parse_args())
