import torch
import torch.nn.functional as F
from torch.distributions import Uniform
import numpy as np
from utils import *
from directories import *
import random
import copy
import torch.nn.functional as nnf
from robustness_measures import softmax_robustness, softmax_difference

DEBUG=False


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
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

        input_shape = image.size(1) * image.size(2) * image.size(3)
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
        adv_examples.append(
            (prediction.item(), new_prediction.item(), perturbed_data.squeeze().detach().to(device).numpy()))

    accuracy = correct / float(len(data_loader.dataset))
    print(
        "\nAttack epsilon = {}\t Accuracy = {} / {} = {}".format(epsilon, correct, len(data_loader.dataset), accuracy))
    return accuracy, adv_examples


def fgsm_bayesian_attack(model, n_attack_samples, n_pred_samples, image, label, epsilon, device):
    image.requires_grad = True
    sum_sign_data_grad = 0.0
    # for i in range(n_samples):
        # sampled_model = model.guide(None, None)
        # output = sampled_model(image)
    original_prediction = model.forward(image, n_samples=n_attack_samples).mean(0)  # .argmax(-1)

    loss = torch.nn.CrossEntropyLoss()(original_prediction, label) # use with softmax

    model.zero_grad()
    loss.backward(retain_graph=True)
    image_grad = image.grad.data
    # Collect the element-wise sign of the data gradient
    sum_sign_data_grad = sum_sign_data_grad + image_grad.sign()

    perturbed_image = image + epsilon * sum_sign_data_grad / n_attack_samples
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image

    perturbation_prediction = model.forward(perturbed_image, n_samples=n_pred_samples).mean(0)

    original_prediction = np.array(original_prediction.cpu().detach())
    perturbation_prediction = np.array(perturbation_prediction.cpu().detach())
    softmax_difference_norm = softmax_difference(original_predictions=original_prediction,
                                                 perturbations_predictions=perturbation_prediction)

    if DEBUG:
        print("true_label =", label.item(),
              "\toriginal_pred =", original_prediction,
              "\tperturbation_pred =", perturbation_prediction,
              "\tsoftmax_diff_norm =", softmax_difference_norm)

    return {"perturbed_image":perturbed_image, "perturbation_prediction":perturbation_prediction,
            "softmax_difference_norm":softmax_difference_norm, "original_prediction":original_prediction}


def attack_bnn(model, n_attack_samples, n_pred_samples, data_loader, epsilon, method="fgsm", device="cpu"):
    # from robustness_measures import min_eps_perturbation

    correct = 0

    pointwise_softmax_differences = []
    pointwise_softmax_robustness = []

    original_predictions = []
    perturbations_predictions = []

    for images, labels in data_loader:
        for idx in range(len(images)):
            image = images[idx]
            label = labels[idx]

            input_shape = image.size(0) * image.size(1) * image.size(2)
            label = label.to(device).argmax(-1).view(-1)
            image = image.to(device).view(-1, input_shape)

            attack = fgsm_bayesian_attack(model=model, n_attack_samples=n_attack_samples, n_pred_samples=n_pred_samples,
                                     image=image, label=label, epsilon=epsilon, device=device)

            # min_eps_pert = min_eps_perturbation(model, n_attack_samples, n_pred_samples, image, label, epsilon, device)

            original_predictions.append(attack["original_prediction"])
            perturbations_predictions.append(attack["perturbation_prediction"])
            pointwise_softmax_differences.append(attack["softmax_difference_norm"])

            if attack["perturbation_prediction"].argmax(-1).item() == label.item():
                correct += 1

    accuracy = correct / float(len(data_loader.dataset)) * 100
    exp_softmax_diff = np.sum(np.array(pointwise_softmax_differences)) / len(data_loader.dataset)
    # softmax_rob = softmax_robustness(np.array(original_predictions), np.array(perturbations_predictions))


    print(f"acc = {accuracy:.2f}", end="\t")
    print(f"exp_softmax_diff = {exp_softmax_diff:.2f}")
    # print("softmax_rob = ", int(softmax_rob), end="\n")

    return {"accuracy":accuracy,
            "pointwise_softmax_differences":pointwise_softmax_differences,
            "pointwise_softmax_robustness":pointwise_softmax_robustness}#, "min_eps_pert":min_eps_pert}
