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
    input_size = image.size(0) * image.size(1) * image.size(2)
    image = image.view(-1, input_size).to(device)
    label = label.to(device).argmax(-1).view(-1)

    if mode == "vi":
        # === old ===
        x = copy.deepcopy(image)
        x.requires_grad = True
        posterior_copy = copy.deepcopy(posterior)
        output = posterior_copy.forward(inputs=x, n_samples=n_samples).to(device)
        avg_output = output.mean(0)
        loss = torch.nn.CrossEntropyLoss()(avg_output, label)
        loss.backward()
        exp_loss_gradient = copy.deepcopy(x.grad.data[0])
        posterior_copy.zero_grad()
        del posterior_copy
        del x

        # === new ===
        # sum_sign_data_grad = 0.0
        # for _ in range(n_samples):
        #     x = copy.deepcopy(image)
        #     x.requires_grad = True
        #     posterior_copy = copy.deepcopy(posterior)
        #     output = posterior_copy.forward(x, n_samples=1).mean(0)
        #     loss = torch.nn.CrossEntropyLoss()(output, label)
        #
        #     posterior_copy.zero_grad()
        #     loss.backward(retain_graph=True)
        #     image_grad = x.grad.data
        #     # Collect the element-wise sign of the data gradient
        #     sum_sign_data_grad = sum_sign_data_grad + image_grad.sign()
        #
        # exp_loss_gradient = sum_sign_data_grad/n_samples
        ##################

    elif mode == "hmc":
        raise NotImplementedError
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
    # mean_over_inputs = exp_loss_gradients.mean(0)  # len = 784
    # std_over_inputs = exp_loss_gradients.std(0)  # len = 784
    # print(f"\nmean_over_inputs[:20] = {mean_over_inputs[:20].cpu().detach()} "
    #       f"\n\nstd_over_inputs[:20] = {std_over_inputs[:20].cpu().detach()}")
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
    return load_from_pickle(path=relpath+str(dataset_name)+"/"+filename).squeeze()

def compute_vanishing_grads_idxs(loss_gradients, n_samples_list):
    if loss_gradients.shape[1] != len(n_samples_list):
        raise ValueError("Second dimension should equal the length of `n_samples_list`")

    vanishing_gradients_idxs = []

    print("\nvanishing gradients norms:")
    count_van_images = 0
    for image_idx, image_gradients in enumerate(loss_gradients):
        # gradient_norm = np.linalg.norm(image_gradients[0])
        gradient_norm = np.max(np.abs(image_gradients[0]))
        if gradient_norm != 0.0:
            print("idx=",image_idx, end="\t\t")
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
                print(", count=", count_van_images)
                count_van_images += 1
            print("\n")

    print("\nvanishing_gradients_idxs = ", vanishing_gradients_idxs)
    return vanishing_gradients_idxs

def compute_constantly_null_grads(loss_gradients, n_samples_list):
    if loss_gradients.shape[1] != len(n_samples_list):
        raise ValueError("Second dimension should equal the length of `n_samples_list`")

    const_null_idxs = []
    for image_idx, image_gradients in enumerate(loss_gradients):
        count_samples_idx = 0
        for samples_idx, n_samples in enumerate(n_samples_list):
            gradient_norm = np.max(np.abs(image_gradients[samples_idx]))
            if gradient_norm == 0.0:
                count_samples_idx += 1
        if count_samples_idx == len(n_samples_list):
            const_null_idxs.append(image_idx)

    print("\nconst_null_idxs = ", const_null_idxs)
    return const_null_idxs

def categorical_loss_gradients_norms(loss_gradients, n_samples_list, dataset_name, model_idx):
    loss_gradients = np.array(np.transpose(loss_gradients, (1,0,2)))

    if loss_gradients.shape[1] != len(n_samples_list):
        raise ValueError("Second dimension should equal the length of `n_samples_list`")

    vanishing_idxs = compute_vanishing_grads_idxs(loss_gradients, n_samples_list)
    const_null_idxs = compute_constantly_null_grads(loss_gradients, n_samples_list)

    loss_gradients_norms_categories = []
    for image_idx in range(len(loss_gradients)):
        if image_idx in vanishing_idxs:
            loss_gradients_norms_categories.append("vanishing")
        elif image_idx in const_null_idxs:
            loss_gradients_norms_categories.append("const_null")
        else:
            loss_gradients_norms_categories.append("other")

    filename = str(dataset_name)+"_bnn_inputs="+str(len(loss_gradients))+\
               "_samples="+str(n_samples_list)+"_cat_lossGrads_norms"+str(model_idx)+".pkl"
    save_to_pickle(data=loss_gradients_norms_categories, relative_path=RESULTS, filename=filename)
    return {"categories":loss_gradients_norms_categories}

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

    # n_inputs, n_samples_list, model_idx, dataset = 100, [500], 2, "mnist"
    # n_inputs, n_samples_list, model_idx, dataset = 1000, [1, 10, 50, 100, 500], 5, "fashion_mnist"

    n_inputs, n_samples_list, model_idx, dataset = 1000, [500], 5, "fashion_mnist"
    model = hidden_vi_models[model_idx]

    # model = {"idx": 6, "filename": "hidden_vi_fashion_mnist_inputs=60000_lr=2e-05_epochs=200", "activation": "softmax",
    # "dataset": "fashion_mnist", "architecture": "fully_connected"} # 75.08 test
    # n_samples_list = [1,5,10,50,100]
    # n_inputs = 1000

    # = compute expected loss gradients =
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=model["dataset"], batch_size=128, n_inputs=n_inputs, shuffle=True)

    for n_samples in n_samples_list:
        bayesnn = VI_BNN(input_shape=input_shape, device=args.device, architecture=model["architecture"],
                         activation=model["activation"])
        posterior = bayesnn.load_posterior(posterior_name=model["filename"], activation=model["activation"],
                                           relative_path=TRAINED_MODELS)

        expected_loss_gradients(posterior=posterior, n_samples=n_samples, dataset_name=model["dataset"],
                                model_idx=model["idx"], data_loader=test_loader, device=args.device, mode="vi")

        del bayesnn, posterior

    # = compute categorical loss gradients =
    # samples_loss_gradients = []
    # for n_samples in n_samples_list:
    #     samples_loss_gradients.append(load_loss_gradients(dataset_name=model["dataset"], n_inputs=n_inputs,
    #                                                       n_samples=n_samples, model_idx=model["idx"]))
    # categorical_loss_gradients_norms(samples_loss_gradients, n_samples_list, dataset_name=model["dataset"],
    #                                  model_idx=model["idx"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cuda', type=str, help='use "cpu" or "cuda".')
    parser.add_argument("--inputs", default=1000, type=int)

    main(args=parser.parse_args())
