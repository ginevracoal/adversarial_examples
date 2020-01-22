import itertools
import sys
sys.path.append(".")
from directories import *
from utils import plot_heatmap
from BayesianInference.pyro_utils import data_loaders, slice_data_loader
import torch
import copy
from utils import save_to_pickle
import numpy as np

DEBUG=True


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

        ## loss = categorical_cross_entropy(y_pred=avg_output, y_true=label)  # use with softmax
        loss = torch.nn.CrossEntropyLoss()(avg_output, label) # use with softmax
        # loss = torch.nn.NLLLoss()(avg_output, label)  # use with log softmax

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


def average_over_images(posterior, n_inputs_list, n_samples_list, device, data_loader, filename, mode="vi"):
    avg_over_images = []
    for n_inputs in n_inputs_list:
        data_loader_slice = slice_data_loader(data_loader=data_loader, slice_size=n_inputs)

        loss_gradients = []
        for n_samples in n_samples_list:
            accuracy = posterior.evaluate(data_loader=data_loader_slice, n_samples=n_samples)
            loss_gradient = expected_loss_gradients(posterior=posterior,
                                                    n_samples=n_samples,
                                                    data_loader=data_loader_slice,
                                                    device=device, mode=mode)

            plot_heatmap(columns=loss_gradient, path=RESULTS + "bnn/",
                         filename="lossGradients_inputs="+str(n_inputs)+"_samples="+str(n_samples)+"_heatmap.png",
                         xlab="pixel idx", ylab="image idx",
                         title=f"Loss gradients pixel components on {n_samples} sampled posteriors")

            avg_loss_gradient = np.mean(np.array(loss_gradient), axis=0)
            loss_gradients.append({"avg_loss_gradient":avg_loss_gradient, "n_samples":n_samples, "n_inputs":n_inputs,
                                   "accuracy":accuracy})
        avg_over_images.append(loss_gradients)

    avg_over_images = np.array(avg_over_images)
    save_to_pickle(data=avg_over_images, relative_path=RESULTS+"bnn/", filename=filename+".pkl")


# === Loss functions ===

# def nllloss(logs, targets):
#     out = torch.zeros_like(targets, dtype=torch.float)
#     for i in range(len(targets)):
#         out[i] = logs[i][targets[i]]
#     return -out.sum()/len(out)


def categorical_cross_entropy(y_pred, y_true):
    # y_pred = predicted probability vector
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    if DEBUG:
        print("\ny_pred = ",y_pred)
        print("\ntorch.log(y_pred) = ",torch.log(y_pred))
        print("\ny_true * torch.log(y_pred) = ",y_true * torch.log(y_pred))
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

