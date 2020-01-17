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

    accuracy = correct / float(len(data_loader))
    print("\nAttack epsilon = {}\t Accuracy = {} / {} = {}".format(epsilon, correct, len(data_loader), accuracy))
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

    accuracy = correct / float(len(data_loader))
    print("\nAttack epsilon = {}\t Accuracy = {} / {} = {}".format(epsilon, correct, len(data_loader), accuracy))
    return accuracy, adv_examples

def categorical_cross_entropy(y_pred, y_true):
    # y_pred = predicted probability vector
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

def expected_loss_gradient(posteriors_list, n_samples, image, label, device, mode):
    random.seed(123)
    loss_gradients = []

    for posterior in posteriors_list:
        for i in range(n_samples):
            x = copy.deepcopy(image)
            x.requires_grad = True
            # print("\ntrue label =", label.item())
            if mode == "vi":
                # here posterior is a bayesnn object which performs random sampling from posterior on forward calls
                output = posterior.forward(inputs=x, n_samples=1).to(device).exp()
                # print("\noutput = ", output.cpu().detach().numpy())
            elif mode == "hmc":
                sampled_model = posterior.posterior_samples[i]
                output = posterior.predict(inputs=x, posterior_samples=[sampled_model])
            else:
                raise ValueError("wrong inference mode")

            output = output.mean(0)

            # print("\noutput = ", output.cpu().detach().numpy())
            # print("\ncheck prob distribution:", output.sum(dim=1).item())#==1.0)

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

    exp_loss_gradient = torch.sum(torch.stack(loss_gradients), dim=0)/(n_samples*len(posteriors_list))
    print("exp_loss_gradient[:5] =", exp_loss_gradient[:5])

    return exp_loss_gradient.cpu().detach().numpy().flatten()


def expected_loss_gradients(posteriors_list, n_samples, data_loader, device, mode):
    print(f"\n === Expected loss gradients on {n_samples} models from {len(posteriors_list)} posteriors"
          f" and {len(data_loader)} input images:")
    expected_loss_gradients = []

    for images, labels in data_loader:
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            # if image.size(0) != 1:
            #     raise ValueError("data_loader batch_size should be 1.")
            input_size = image.size(0) * image.size(1) * image.size(2)
            label = label.to(device).argmax(-1).to(device)
            image = image.to(device).view(-1, input_size).to(device)
            expected_loss_gradients.append(expected_loss_gradient(posteriors_list, n_samples, image, label, mode=mode,
                                                                  device=device))

    np_exp_loss_gradients = np.array(expected_loss_gradients)

    # summary statistics along input points
    mean_over_inputs = np.mean(np_exp_loss_gradients, axis=0) # len = 784
    std_over_inputs = np.std(np_exp_loss_gradients, axis=0) # len = 784
    print(f"\nmean_over_inputs[:20] = {mean_over_inputs[:20]} \n\nstd_over_inputs[:20] = {std_over_inputs[:20]} \t")

    filename = "expLossGradients_samples="+str(n_samples)+"_inputs="+str(len(np_exp_loss_gradients))\
               +"_posteriors="+str(len(posteriors_list))
    # save_to_pickle(np_exp_loss_gradients, relative_path=RESULTS+"bnn/", filename=filename+".pkl")
    return np_exp_loss_gradients


def plot_expectation_over_images(dataset_name, n_inputs, n_samples_list, rel_path=RESULTS):

    avg_loss_gradients = []
    for n_samples in n_samples_list:
        filename = "expLossGradients_samples="+str(n_samples)+"_inputs="+str(n_inputs)
        expected_loss_gradients = load_from_pickle(path=rel_path+"bnn/"+filename+".pkl")
        avg_loss_gradient = np.mean(expected_loss_gradients, axis=0)/n_inputs
        avg_loss_gradients.append(avg_loss_gradient)
        print("\nn_samples={} \navg_loss_gradient[:10]={}".format(n_samples,avg_loss_gradient[:10]))

    filename = "hidden_vi_" + str(dataset_name) + "_inputs=" + str(n_inputs)
    plot_heatmap(columns=avg_loss_gradients, path=RESULTS + "bnn/", filename=filename+"_heatmap.png",
                 xlab="pixel idx", ylab="n. posterior samples", yticks=n_samples_list,
                 title="Expected loss gradients over {} images".format(n_inputs))
