import sys
sys.path.append(".")
from directories import *
import argparse
import pyro
import pandas as pd
import matplotlib
from robustness_measures import softmax_difference
from BayesianInference.adversarial_attacks import *
from BayesianInference.loss_gradients import load_loss_gradients, categorical_loss_gradients_norms

DATA_PATH="../data/random_attack_plot/"


def plot_random_attack(samples, softmax_differences, attack_type):
    raise NotImplementedError

def attack_model(model, loss_gradients, test_loader, n_samples_list, device):
    eps = 0.25

    samples = []
    softmax_diff = []
    attack_type = []

    im_count = 0
    for n_samples in n_samples_list:
        print("\nn_samples = ", n_samples)
        for images, labels in test_loader:
            for idx in tqdm(range(len(images))):
                image = images[idx]
                label = labels[idx]

                input_shape = image.size(0) * image.size(1) * image.size(2)
                label = label.to(device).argmax(-1).view(-1)
                image = image.to(device).view(-1, input_shape)

                # random attack
                attack_dict = random_bayesian_attack(model=model, image=image,
                                                     label=label, epsilon=eps, device=device, n_pred_samples=n_samples)


                original_output = attack_dict["original_output"].clone().detach().to(device)
                adversarial_output = attack_dict["adversarial_output"].clone().detach().to(device)
                softmax_diff.append((original_output - adversarial_output).abs().max(dim=-1)[0])
                samples.append(n_samples)
                attack_type.append("fgsm")

                # bayesian fgsm attack
                loss_gradient = torch.tensor(loss_gradients[str(n_samples)][im_count])
                attack_dict = fgsm_bayesian_attack(model=model, image=image,
                                                  label=label, epsilon=eps, device=device, n_pred_samples=n_samples,
                                                   n_attack_samples=n_samples,
                                                   loss_gradient=loss_gradient)
                original_output = attack_dict["original_output"].clone().detach().to(device)
                adversarial_output = attack_dict["adversarial_output"].clone().detach().to(device)
                softmax_diff.append((original_output - adversarial_output).abs().max(dim=-1)[0])
                samples.append(n_samples)
                attack_type.append("fgsm")

    random_attack_dict = {"samples":samples, "softmax_diff":softmax_diff, "attack_type":attack_type}
    return random_attack_dict

def load_attacks():
    raise NotImplementedError

def main(args):
    # == choose model ==
    # n_inputs, n_samples_list, model_idx, dataset, relpath = 1000, [1,10,50,100], 2, "mnist", DATA_PATH
    n_inputs, n_samples_list, model_idx, dataset, relpath = 1000, [1,10,50,100], 5, "fashion_mnist", DATA_PATH

    # == load data ==
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset, batch_size=128, n_inputs=n_inputs, shuffle=True)
    test = slice_data_loader(data_loader=test_loader, slice_size=args.inputs)

    # == load posterior predictive ==
    model = hidden_vi_models[model_idx]
    bayesnn = VI_BNN(input_shape=input_shape, device=args.device, architecture=model["architecture"],
                     activation=model["activation"])
    posterior = bayesnn.load_posterior(posterior_name=model["filename"], relative_path=TRAINED_MODELS,
                                       activation=model["activation"])

    # == load loss gradients ==
    loss_gradients = {}
    for n_samples in n_samples_list:
        gradients = load_loss_gradients(dataset_name=dataset, n_inputs=n_inputs, n_samples=n_samples, model_idx=model_idx)
        loss_gradients.update({str(n_samples):gradients})

    # == attack and plot ==
    random_attack_dict = attack_model(model=posterior, loss_gradients=loss_gradients, test_loader=test,
                                      n_samples_list=n_samples_list, device=args.device)
    filename = str(dataset)+"_inputs="+str(args.inputs)+"_samples="+str(n_samples_list)+"_random_attack.pkl"
    save_to_pickle(data=random_attack_dict, relative_path=RESULTS+"attacks/", filename=filename)
    # random_attack_dict=load_from_pickle(RESULTS+"attacks/"+filename)

    # plot_random_attack(attack_type=random_attack_dict["attack_type"], samples=random_attack_dict["samples"],
    #                    softmax_differences=random_attack_dict["softmax_differences"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--inputs", default=100, type=int)
    main(args=parser.parse_args())

