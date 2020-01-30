import sys
sys.path.append(".")
from directories import *

from tqdm import tqdm
from BayesianInference.pyro_utils import data_loaders, slice_data_loader
import torch
from BayesianInference.hidden_vi_bnn import VI_BNN, hidden_vi_models
import copy
from utils import save_to_pickle, load_from_pickle
import numpy as np
import argparse


DEBUG=False
DATA_PATH="../data/exp_loss_gradients/"


def get_filename(dataset_name, n_inputs, n_samples, model_idx):
    return str(dataset_name)+"_inputs="+str(n_inputs)+"_samples="+str(n_samples)+"_loss_grads_"+str(model_idx)+".pkl"

def expected_loss_gradient(posterior, n_samples, image, label, device, mode, baseclass=None):
    loss_gradients = []
    input_size = image.size(0) * image.size(1) * image.size(2)
    image = image.view(-1, input_size).to(device)
    label = label.to(device).argmax(-1).view(-1)

    if mode == "vi":
        x = copy.deepcopy(image)
        x.requires_grad = True
        posterior_copy = copy.deepcopy(posterior)
        # posterior is a bayesnn object which performs random sampling from n_samples posteriors on forward calls
        log_output = posterior_copy.forward(inputs=x, n_samples=n_samples).to(device)
        avg_output = log_output.mean(0)

        if DEBUG:
            print("\ntrue label =", label.item())
            # print("\nlog_output[:5]=", log_output[:5].cpu().detach().numpy())
            print("\nlog_output shape=", log_output.shape)
            # print("\noutput=", output[:10].cpu().detach().numpy())
            print("avg_output shape=",avg_output.shape)
            print("avg_output[:5]=",avg_output[:5].cpu().detach().numpy())
            # print("\navg_output.exp() =", avg_output.exp().cpu().detach().numpy())
            print("check prob distribution:", avg_output.sum(dim=1).item())
            # print("\ncheck prob distribution:", avg_output.exp().sum(dim=1).item())

        if posterior.loss == "crossentropy":
            loss = torch.nn.CrossEntropyLoss()(avg_output, label)  # use with softmax
        elif posterior.loss == "nllloss":
            # loss = torch.nn.NLLLoss()(avg_output, label)  # use with log softmax
            loss = torch.nn.CrossEntropyLoss()(avg_output.exp(), label)  # use with log softmax
        else:
            raise AttributeError("Wrong loss function.")

        loss.backward()
        loss_gradient = copy.deepcopy(x.grad.data[0])
        loss_gradients.append(loss_gradient)
        if DEBUG:
            print("\nloss = ", loss.item())
            print("\nloss_gradient[:5] = ", loss_gradient[:5].cpu().detach().numpy()) # len = 784
        posterior_copy.zero_grad()
        del posterior_copy
        del x

    # elif mode == "hmc":
    #     x = copy.deepcopy(image)
    #     x.requires_grad = True
    #     output = baseclass.predict(inputs=x, posterior_samples=posteriors)
    #     # print("\noutput = ", output.cpu().detach().numpy())
    #     # print("\ncheck prob distribution:", output.sum(dim=1).item())
    #     loss = categorical_cross_entropy(y_pred=output, y_true=label)
    #
    #     loss.backward()
    #     loss_gradient = copy.deepcopy(x.grad.data[0])
    #     loss_gradients.append(loss_gradient)
    #     del x

    else:
        raise ValueError("wrong inference mode")

    exp_loss_gradient = torch.stack(loss_gradients).mean(dim=0)
    # print("\nexp_loss_gradient[:20] =", exp_loss_gradient[:20])

    # print(f"mean_over_features = {exp_loss_gradient.mean(0).item()} "
    #       f"\tstd_over_features = {exp_loss_gradient.std(0).item()}")

    return exp_loss_gradient


def expected_loss_gradients(posterior, n_samples, data_loader, dataset_name, device, mode, model_idx, baseclass=None):
    print(f"\n === Expected loss gradients on {n_samples} posteriors"
          f" and {len(data_loader.dataset)} input images:")
    exp_loss_gradients = []

    for images, labels in data_loader:
        for i in tqdm(range(len(images))):
            exp_loss_gradients.append(expected_loss_gradient(posterior=posterior, n_samples=n_samples, image=images[i],
                                                             label=labels[i], mode=mode, device=device))

    exp_loss_gradients = torch.stack(exp_loss_gradients)
    mean_over_inputs = exp_loss_gradients.mean(0)  # len = 784
    std_over_inputs = exp_loss_gradients.std(0)  # len = 784

    print(f"\nmean_over_inputs[:20] = {mean_over_inputs[:20].cpu().detach()} "
          f"\n\nstd_over_inputs[:20] = {std_over_inputs[:20].cpu().detach()}")
    print(f"\nexp_overall_mean = {exp_loss_gradients.mean()} \t exp_overall_std = {exp_loss_gradients.std()}")

    exp_loss_gradients = exp_loss_gradients.cpu().detach().numpy()

    filename = get_filename(dataset_name=dataset_name, n_inputs=len(data_loader.dataset), n_samples=n_samples,
                            model_idx=model_idx)
    save_to_pickle(data=exp_loss_gradients, relative_path=RESULTS+str(dataset_name)+"/", filename=filename)
    return exp_loss_gradients


def expected_loss_gradients_multiple_posteriors(posteriors_list, n_samples, data_loader, device, mode, baseclass=None):
    for posterior in posteriors_list:
        expected_loss_gradients(posterior, n_samples, data_loader, device, mode, baseclass=None, mode=mode)


def categorical_cross_entropy(y_pred, y_true):
    # y_pred = predicted probability vector
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    if DEBUG:
        print("\ny_pred = ",y_pred)
        print("\ntorch.log(y_pred) = ",torch.log(y_pred))
        print("\ny_true * torch.log(y_pred) = ",y_true * torch.log(y_pred))
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()


def load_loss_gradients(dataset_name, n_inputs, n_samples, model_idx, relpath=DATA_PATH):
    filename = get_filename(dataset_name, n_inputs, n_samples, model_idx)
    return load_from_pickle(path=relpath+str(dataset_name)+"/"+filename)


def compute_vanishing_grads_idxs(loss_gradients, n_samples_list):
    if loss_gradients.shape[1] != len(n_samples_list):
        raise ValueError("Second dimension should equal the length of `n_samples_list`")

    vanishing_gradients_idxs = []

    print("\nvanishing gradients norms:")
    for image_idx, image_gradients in enumerate(loss_gradients):
        # gradient_norm = np.linalg.norm(image_gradients[0])
        gradient_norm = np.max(np.abs(image_gradients[0]))
        if gradient_norm != 0.0:
            count_samples_idx = 0
            for samples_idx, n_samples in enumerate(n_samples_list):
                # new_gradient_norm = np.linalg.norm(image_gradients[samples_idx])
                new_gradient_norm = np.max(np.abs(image_gradients[samples_idx]))
                if new_gradient_norm <= gradient_norm:
                    print(new_gradient_norm, end="\t")
                    gradient_norm = copy.deepcopy(new_gradient_norm)
                    count_samples_idx += 1
            if count_samples_idx == len(n_samples_list):
                vanishing_gradients_idxs.append(image_idx)
                print("\n")

    print("\nvanishing_gradients_idxs = ", vanishing_gradients_idxs)
    return vanishing_gradients_idxs


# # todo: soon deprecated
# def load_multiple_loss_gradients_old(dataset_name, n_inputs, eps, model_idx, relpath=DATA_PATH):
#
#     exp_loss_gradients = []
#
#     if n_inputs == 100 and dataset_name=="mnist" and model_idx in [0,1,2] and eps in [0.1,0.3,0.6]:
#
#         n_samples_list = [1, 10, 50, 100]
#         for samples in n_samples_list:
#             filename = "mnist_inputs=100_epsilon="+str(eps)+"_samples="+str(samples)+"_model="+str(model_idx)+"_attack.pkl"
#             exp_loss_gradients.append(load_from_pickle(path=relpath+str(dataset_name)+"/"+filename))
#
#     elif n_inputs == 10 and dataset_name=="mnist" and model_idx in [0,1,2]:
#
#         n_samples_list = [1, 10, 50, 100, 500]
#         for samples in [1, 10, 50, 100]:
#             filename = "mnist_inputs=100_epsilon=0.1_samples="+ str(samples)+ "_model=" + str(model_idx) + "_attack.pkl"
#             exp_loss_gradients.append(load_from_pickle(path=relpath+str(dataset_name)+"/"+filename)[:10])
#         filename = "mnist_inputs=10_epsilon=0.1_samples=500_model=" + str(model_idx) + "_attack.pkl"
#         exp_loss_gradients.append(load_from_pickle(path=relpath+str(dataset_name)+"/"+filename))
#
#     elif n_inputs == 1000 and dataset_name == "mnist" and model_idx == 2:
#
#         n_samples_list = [1, 10, 50, 100]
#         for samples in [1, 10]:
#             filename = "mnist_inputs=1000_epsilon=0.1_samples="+str(samples)+"_model="+str(model_idx)+"_attack.pkl"
#             exp_loss_gradients.append(load_from_pickle(path=relpath+str(dataset_name)+"/"+filename))
#         for samples in [50, 100]:
#             filename = "mnist_inputs=1000_samples="+str(samples)+"_model="+str(model_idx)+"_loss_gradients.pkl"
#             exp_loss_gradients.append(load_from_pickle(path=relpath+str(dataset_name)+"/"+filename))
#     else:
#         raise AssertionError("loss gradients are not available for the chosen params.")
#
#     exp_loss_gradients = np.array(exp_loss_gradients)
#     print("exp_loss_gradients.shape =", exp_loss_gradients.shape)
#     return exp_loss_gradients, n_samples_list
###############

def main(args):

    # n_inputs, n_samples_list, model_idx, dataset = 1000, [1,5,10,50,100], 2, "mnist"
    n_inputs, n_samples_list, model_idx, dataset = 1000, [1,5,10,50,100], 5, "fashion_mnist"


    model = hidden_vi_models[model_idx]
    for n_samples in n_samples_list:
        train_loader, test_loader, data_format, input_shape = \
            data_loaders(dataset_name=model["dataset"], batch_size=128, n_inputs=n_inputs, shuffle=True)

        bayesnn = VI_BNN(input_shape=input_shape, device=args.device, architecture=model["architecture"],
                         activation=model["activation"])
        posterior = bayesnn.load_posterior(posterior_name=model["filename"],
                                           relative_path=TRAINED_MODELS,
                                           activation=model["activation"])

        expected_loss_gradients(posterior=posterior, n_samples=n_samples, dataset_name=dataset,
                                model_idx=model["idx"], data_loader=test_loader, device=args.device, mode="vi")

        del bayesnn, posterior


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--inputs", default=1000, type=int)

    main(args=parser.parse_args())
