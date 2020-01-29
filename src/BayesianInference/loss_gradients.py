import sys
sys.path.append(".")
from directories import *
from utils import plot_heatmap
from BayesianInference.pyro_utils import data_loaders, slice_data_loader
import torch
from BayesianInference.hidden_vi_bnn import VI_BNN, hidden_vi_models
import copy
from utils import save_to_pickle, load_from_pickle
import numpy as np
import argparse


DEBUG=False


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

    print(f"mean_over_features = {exp_loss_gradient.mean(0).item()} "
          f"\tstd_over_features = {exp_loss_gradient.std(0).item()}")

    return exp_loss_gradient


def expected_loss_gradients(posterior, n_samples, data_loader, device, mode, baseclass=None):
    print(f"\n === Expected loss gradients on {n_samples} posteriors"
          f" and {len(data_loader.dataset)} input images:")
    exp_loss_gradients = []

    for images, labels in data_loader:
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            exp_loss_gradients.append(expected_loss_gradient(posterior=posterior, n_samples=n_samples, image=image,
                                                            label=label, mode=mode, device=device))

    exp_loss_gradients = torch.stack(exp_loss_gradients)
    mean_over_inputs = exp_loss_gradients.mean(0)  # len = 784
    std_over_inputs = exp_loss_gradients.std(0)  # len = 784

    print(f"\nmean_over_inputs[:20] = {mean_over_inputs[:20].cpu().detach()} "
          f"\n\nstd_over_inputs[:20] = {std_over_inputs[:20].cpu().detach()}")

    print(f"\nexp_mean = {exp_loss_gradients.mean()} \t exp_std = {exp_loss_gradients.std()}")

    return exp_loss_gradients.cpu().detach().numpy()


def expected_loss_gradients_multiple_posteriors(posteriors_list, n_samples, data_loader, device, mode, baseclass=None):
    for posterior in posteriors_list:
        expected_loss_gradients(posterior, n_samples, data_loader, device, mode, baseclass=None)

def categorical_cross_entropy(y_pred, y_true):
    # y_pred = predicted probability vector
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    if DEBUG:
        print("\ny_pred = ",y_pred)
        print("\ntorch.log(y_pred) = ",torch.log(y_pred))
        print("\ny_true * torch.log(y_pred) = ",y_true * torch.log(y_pred))
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()


def load_loss_gradients(filename, relpath="../data/exp_loss_gradients/"):
    """ Loads any pkl dictionary containing the key `loss_gradients`. """

    attack_dict = load_from_pickle(relpath + filename)
    loss_gradients = np.array([np.array(gradient) for gradient in attack_dict["loss_gradients"]])
    loss_gradients = np.squeeze(loss_gradients)
    # print("\nloss_gradients.shape = ", loss_gradients.shape)
    return loss_gradients


def load_multiple_loss_gradients(dataset_name, n_inputs, eps, model_idx, relpath="../data/exp_loss_gradients/"):

    exp_loss_gradients = []

    if n_inputs == 100 and dataset_name=="mnist" and model_idx in [0,1,2] and eps in [0.1,0.3,0.6]:

        n_samples_list = [1, 10, 50, 100]
        for samples in n_samples_list:
            filename = "mnist_inputs=100_epsilon="+str(eps)+"_samples="+str(samples)+"_model="+str(model_idx)+"_attack.pkl"
            exp_loss_gradients.append(load_loss_gradients(filename=filename, relpath=relpath))

    elif n_inputs == 10 and dataset_name=="mnist" and model_idx in [0,1,2]:

        n_samples_list = [1, 10, 50, 100, 500]
        for samples in [1, 10, 50, 100]:
            filename = "mnist_inputs=100_epsilon=0.1_samples="+ str(samples)+ "_model=" + str(model_idx) + "_attack.pkl"
            exp_loss_gradients.append(load_loss_gradients(filename=filename, relpath=relpath)[:10])
        filename = "mnist_inputs=10_epsilon=0.1_samples=500_model=" + str(model_idx) + "_attack.pkl"
        exp_loss_gradients.append(load_loss_gradients(filename=filename, relpath=relpath))

    elif n_inputs == 1000 and dataset_name == "mnist" and model_idx == 2:

        n_samples_list = [1, 10, 50, 100]
        for samples in [1, 10]:
            filename = "mnist_inputs=1000_epsilon=0.1_samples="+str(samples)+"_model="+str(model_idx)+"_attack.pkl"
            exp_loss_gradients.append(load_loss_gradients(filename=filename, relpath=relpath))
        for samples in [50, 100]:
            filename = "mnist_inputs=1000_samples="+str(samples)+"_model="+str(model_idx)+"_loss_gradients.pkl"
            exp_loss_gradients.append(load_loss_gradients(filename=filename, relpath=relpath))
    else:
        raise AssertionError("loss gradients are not available for the chosen params.")

    exp_loss_gradients = np.array(exp_loss_gradients)
    print("exp_loss_gradients.shape =", exp_loss_gradients.shape)
    return exp_loss_gradients, n_samples_list


def main(args):
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=args.dataset, batch_size=36, n_inputs=args.inputs, shuffle=True)

    model = hidden_vi_models[3]
    n_samples_list = [1,10,50,100]

    for n_samples in n_samples_list:
        bayesnn = VI_BNN(input_shape=input_shape, device=args.device, architecture=model["architecture"],
                         activation=model["activation"])
        posterior = bayesnn.load_posterior(posterior_name=model["filename"],
                                           relative_path=TRAINED_MODELS,
                                           activation=model["activation"])

        exp_loss_gradients = expected_loss_gradients(posterior=posterior, n_samples=n_samples,
                                                     data_loader=train_loader, device=args.device, mode="vi")

        filename = args.dataset + "_inputs=" + str(args.inputs) \
                   + "_samples=" + str(n_samples) + "_model=" + str(model["idx"]) + "_loss_gradients.pkl"
        loss_gradients_dict = {"loss_gradients": exp_loss_gradients}
        save_to_pickle(data=loss_gradients_dict, relative_path=RESULTS + "bnn/", filename=filename)

        del bayesnn, posterior, loss_gradients_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--inputs", default=100, type=int)

    main(args=parser.parse_args())
