import torch
import torch.nn.functional as F
from torch.distributions import Uniform
import numpy as np
from utils import *
from directories import *
import random
import copy
import torch.nn.functional as nnf


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def attack_nn(model, data_loader, method="fgsm", device="cpu"):
    correct = 0
    adv_examples = []
    epsilon = Uniform(0.2, 0.4).sample()

    for image, label in data_loader:
        if image.size(0) != 1:
            raise ValueError("data_loader batch_size should be 1.")

        input_shape = image.size(1)*image.size(2)*image.size(3)
        label = label.to(device).argmax(-1)
        image = image.to(device).view(-1, input_shape)

        image.requires_grad = True
        output = model(image)
        # index of the max log-probability
        prediction = output.max(1, keepdim=True)[1]

        # if prediction is wrong move on
        if prediction.item() != label.item():
            continue

        # loss
        loss = F.cross_entropy(output, label)

        # zero gradients
        model.zero_grad()
        # compute gradients
        loss.backward(retain_graph=True)
        image_grad = image.grad.data
        perturbed_data = fgsm_attack(image, epsilon, image_grad)
        new_output = model(perturbed_data)
        new_prediction = new_output.max(1, keepdim=True)[1]
        if new_prediction.item() == label.item():
            correct += 1
        adv_examples.append((prediction.item(), new_prediction.item(), perturbed_data.squeeze().detach().to(device).numpy()))

    accuracy = correct / float(len(data_loader.dataset))
    print("\nAttack epsilon = {}\t Accuracy = {} / {} = {}".format(epsilon, correct, len(data_loader.dataset), accuracy))
    return accuracy, adv_examples


def fgsm_bayesian_attack(model, n_samples, image, label, epsilon, device):
    image.requires_grad = True
    sum_sign_data_grad = 0.0
    for i in range(n_samples):
        # sampled_model = model.guide(None, None)
        # output = sampled_model(image)
        output = model.forward(image.to(device), n_samples=1).mean(0)#.argmax(-1)
        # print(output)
        # exit()
        loss = F.cross_entropy(output, label)
        # zero gradients
        model.zero_grad()
        # compute gradients
        loss.backward(retain_graph=True)
        image_grad = image.grad.data
        # Collect the element-wise sign of the data gradient
        sum_sign_data_grad = sum_sign_data_grad + image_grad.sign()

    perturbed_image = image + epsilon * sum_sign_data_grad / n_samples
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def attack_bnn(model, n_samples, data_loader, method="fgsm", device="cpu"):
    correct = 0
    adv_examples = []
    epsilon = Uniform(0.2, 0.4).sample()

    for image, label in data_loader:
        if image.size(0) != 1:
            raise ValueError("data_loader batch_size should be 1.")

        input_shape = image.size(1)*image.size(2)*image.size(3)
        label = label.to(device).argmax(-1)
        image = image.to(device).view(-1, input_shape)

        perturbed_image = fgsm_bayesian_attack(model, n_samples, image, label, epsilon, device=device)
        prediction = model.forward(perturbed_image)
        if prediction.item() == label.item():
            correct += 1
        adv_examples.append((prediction.item(), prediction.item(), perturbed_image.squeeze().detach().to(device).numpy()))

    accuracy = correct / float(len(data_loader.dataset))
    print("\nAttack epsilon = {}\t Accuracy = {} / {} = {}".format(epsilon, correct, len(data_loader.dataset), accuracy))
    return accuracy, adv_examples


def categorical_cross_entropy(y_pred, y_true):
    # y_pred = predicted probability vector
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()


def expected_loss_gradient(posteriors, n_samples, image, label, device, mode, baseclass=None):
    random.seed(123)
    loss_gradients = []

    if mode == "vi":
        # qua posteriors è una lista di posterior distributions
        for posterior in posteriors:
            x = copy.deepcopy(image)
            x.requires_grad = True
            # print("\ntrue label =", label.item())

            # posterior is a bayesnn object which performs random sampling from n_samples posteriors on forward calls
            output = posterior.forward(inputs=x, n_samples=n_samples).to(device).exp().mean(0)

            # print("\noutput = ", output.cpu().detach().numpy())
            # print("\ncheck prob distribution:", output.sum(dim=1).item())

            # loss = torch.nn.CrossEntropyLoss()(output, label)
            # loss = F.cross_entropy(output, label)
            # loss = torch.nn.NLLLoss()(output.exp(), label)
            loss = categorical_cross_entropy(y_pred=output, y_true=label)

            loss.backward()
            loss_gradient = copy.deepcopy(x.grad.data[0])
            loss_gradients.append(loss_gradient)
            # print("\nloss = ", loss.item())
            # print("loss_gradient[:5] = ", loss_gradient[:5].cpu().detach().numpy()) # len = 784
            # exit()
            posterior.zero_grad()
            del x

    elif mode == "hmc":
        x = copy.deepcopy(image)
        x.requires_grad = True
        output = baseclass.predict(inputs=x, posterior_samples=posteriors)
        # print("\noutput = ", output.cpu().detach().numpy())
        # print("\ncheck prob distribution:", output.sum(dim=1).item())
        loss = categorical_cross_entropy(y_pred=output, y_true=label)

        loss.backward()
        loss_gradient = copy.deepcopy(x.grad.data[0])
        loss_gradients.append(loss_gradient)
        del x

    else:
        raise ValueError("wrong inference mode")

    exp_loss_gradient = torch.stack(loss_gradients).mean(dim=0)

    # print("\nexp_loss_gradient[:20] =", exp_loss_gradient[:20])
    print(f"mean_over_features = {exp_loss_gradient.mean(0).item()} "
          f"\tstd_over_features = {exp_loss_gradient.std(0).item()}")

    return exp_loss_gradient  #.cpu().detach().numpy().flatten()


def expected_loss_gradients(posteriors, n_samples, data_loader, device, mode, baseclass=None):
    print(f"\n === Expected loss gradients on {n_samples*len(posteriors)} posteriors"
          f" and {len(data_loader.dataset)} input images:")
    exp_loss_gradients = []

    for images, labels in data_loader:
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            input_size = image.size(0) * image.size(1) * image.size(2)
            label = label.to(device).argmax(-1).to(device)
            image = image.to(device).view(-1, input_size).to(device)
            exp_loss_gradients.append(expected_loss_gradient(posteriors=posteriors, n_samples=n_samples, image=image,
                                                             label=label, mode=mode, device=device, baseclass=baseclass))

    # summary statistics over input points
    # exp_loss_gradients = np.array(exp_loss_gradients)
    # mean_over_inputs = np.mean(np_exp_loss_gradients, axis=0)  # len = 784
    # std_over_inputs = np.std(np_exp_loss_gradients, axis=0)  # len = 784

    exp_loss_gradients = torch.stack(exp_loss_gradients)
    mean_over_inputs = exp_loss_gradients.mean(0)  # len = 784
    std_over_inputs = exp_loss_gradients.std(0)  # len = 784

    print(f"\nmean_over_inputs[:20] = {mean_over_inputs[:20].cpu().detach().flatten()} "
          f"\n\nstd_over_inputs[:20] = {std_over_inputs[:20].cpu().detach().flatten()}")
    print(f"\nexp_mean = {exp_loss_gradients.mean()} \t exp_std = {exp_loss_gradients.std()}")

    # filename = "expLossGradients_inputs="+str(len(data_loader.dataset))\
    #            +"_samples="+str(len(posteriors)*n_samples)+"_mode="+str(mode)+".pkl"
    # save_to_pickle(exp_loss_gradients, relative_path=RESULTS+"bnn/", filename=filename)

    return exp_loss_gradients.cpu().detach().numpy()


# PLOT FUNCTIONS


def plot_expectation_over_images(dataset_name, n_inputs, n_samples_list, rel_path=RESULTS):

    avg_loss_gradients = []
    for n_samples in n_samples_list:
        filename = "expLossGradients_samples="+str(n_samples)+"_inputs="+str(n_inputs)
        expected_loss_gradients = load_from_pickle(path=rel_path+"bnn/"+filename+".pkl")
        avg_loss_gradient = np.mean(expected_loss_gradients, axis=0)/n_inputs
        avg_loss_gradients.append(avg_loss_gradient)
        print("\nn_samples={} \navg_loss_gradient[:10]={}".format(n_samples,avg_loss_gradient[:10]))

    filename = "hidden_vi_" + str(dataset_name) + "_inputs=" + str(n_inputs)
    plot_heatmap(columns=avg_loss_gradients, path=RESULTS, filename=filename+"_heatmap.png",
                 xlab="pixel idx", ylab="n. posterior samples", yticks=n_samples_list,
                 title="Expected loss gradients over {} images".format(n_inputs))


def plot_exp_loss_gradients_norms(dataset_name, n_inputs, n_samples_list, n_posteriors, pnorm=2, rel_path=RESULTS):

    exp_loss_gradients_norms = []
    for n_samples in n_samples_list:
        filename = "expLossGradients_samples="+str(n_samples)+"_inputs="+str(n_inputs)\
                    +"_posteriors="+str(n_posteriors) +".pkl"
        exp_loss_gradients = load_from_pickle(path=rel_path + "bnn/" + filename)
        norms = [torch.norm(p=pnorm, input=gradient).item() for gradient in exp_loss_gradients]
        print("\nnorms = ", norms)
        exp_loss_gradients_norms.append(norms)

    filename = "exp_loss_gradients_norms_inputs="+str(n_inputs)+"_posteriors="+str(n_posteriors)+\
               "_pnorm="+str(pnorm)+"_heatmap.png"
    yticks = [n_samples*n_posteriors for n_samples in n_samples_list]
    plot_heatmap(columns=exp_loss_gradients_norms, path=RESULTS, filename=filename,
                 xlab="image idx", ylab="n. posterior samples", yticks=yticks,
                 title="Expected loss gradients norms on {} images ".format(n_inputs))

def plot_partial_derivatives(dataset_name, n_inputs, n_samples_list, n_posteriors=1, rel_path=RESULTS):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    exp_loss_grad = []
    n_samples_column = []
    for i, n_samples in enumerate(n_samples_list):
        filename = "expLossGradients_samples="+str(n_samples)+"_inputs="+str(n_inputs)\
                    +"_posteriors="+str(n_posteriors) +".pkl"
        loss_gradients = load_from_pickle(path=rel_path + "bnn/" + filename)
        avg_partial_derivatives = loss_gradients.mean(0).log().cpu().detach().numpy() #.log()
        # for j in range(loss_gradients.shape[0]): # images
        for k in range(len(avg_partial_derivatives)): # partial derivatives
            exp_loss_grad.append(avg_partial_derivatives[k])
            n_samples_column.append(n_samples*n_posteriors)

    df = pd.DataFrame(data={"log(loss partial derivatives)":exp_loss_grad,"n_samples":n_samples_column})
    print(df.head())
    # print(df.describe(include='all'))

    filename = "partial_derivatives_inputs=" + str(n_inputs)  \
               + "_posteriors=" + str(n_posteriors) + "_catplot.png"

    plot = sns.catplot(data=df, y="log(loss partial derivatives)", x="n_samples", kind="boxen")
    plot.fig.set_figheight(8)
    plot.fig.set_figwidth(15)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plot.savefig(RESULTS + filename, dpi=100)
