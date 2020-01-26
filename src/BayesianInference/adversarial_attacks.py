import torch
import torch.nn.functional as F
from torch.distributions import Uniform
import numpy as np

from BayesianInference.plot_utils import distplot_pointwise_softmax_differences
from utils import *
from directories import *
import random
import copy
import torch.nn.functional as nnf
from robustness_measures import softmax_robustness, softmax_difference
import pandas

DEBUG=False


# def attack_statistics(data_loader, model, epsilon, device, model_idx):
#     """ Attacks a classical nn with epsilon perturbations of the inputs.
#     :returns : "original_accuracy", "adversarial_accuracy", "softmax_robustness", "loss_gradients"
#     """
#
#     original_accuracy, original_predictions  = model.evaluate(data_loader=data_loader, device=device)
#
#     attacks = attack(model=model, data_loader=data, epsilon=epsilon, device=device)
#
#         for i in range(len(data.dataset)):
#             pointwise_softmax_robustness.append(attacks["softmax_robustness"][i])
#             plot_eps.append(epsilon)
#             plot_original_acc.append(original_acc)
#
#     softmax_robustness(original_predictions=)
#
#     dict = {"attacks": attacks, "original_accuracy":original_acc,
#             "adversarial_accuracy":adversarial_acc, "softmax_robustness": softmax_robustness}
#         df = pandas.DataFrame(data=)
#         df.to_pickle(RESULTS + "bnn/" + "pointwise_softmax_differences_eps=" + str(epsilon) \
#                      + "_inputs=" + str(len(data_loader.dataset))
#                      + "_mode=vi_model=" + str(model_idx) + ".pkl")

        # distplot_pointwise_softmax_robustness(df, n_inputs=len(data_loader.dataset),
        #                                        epsilon=epsilon,
        #                                        model_idx=model_idx)

# todo refactor
def pointwise_bayesian_attacks(data_loader, epsilon_list, n_samples_list, posterior, device, idx):
    data = data_loader
    plot_samples = []
    plot_softmax_differences = []
    plot_original_acc = []
    plot_eps = []

    for epsilon in epsilon_list:
        print("\n\nepsilon =", epsilon)

        for n_attack_samples in n_samples_list:
            # print("n_samples =", n_attack_samples, end="\t")
            original_acc = posterior.evaluate(data_loader=data, n_samples=n_attack_samples, device=device)
            attacks = bayesian_attack(model=posterior, n_pred_samples=n_attack_samples,
                                      n_attack_samples=n_attack_samples, data_loader=data, epsilon=epsilon,
                                      device=device)

            for i in range(len(data.dataset)):
                plot_softmax_differences.append(attacks["pointwise_softmax_differences"][i])
                plot_samples.append(n_attack_samples)
                plot_eps.append(epsilon)
                plot_original_acc.append(original_acc)

        df = pandas.DataFrame(data={"n_samples": plot_samples, "softmax_differences": plot_softmax_differences,
                                    "accuracy": plot_original_acc})#, "epsilon": plot_eps})
        df.to_pickle(RESULTS + "bnn/" + "pointwise_softmax_differences_eps=" + str(epsilon) \
               + "_inputs=" + str(len(data_loader.dataset)) + "_samples="+str(n_samples_list)
                     +"_mode=vi_model=" + str(idx) + ".pkl")
        distplot_pointwise_softmax_differences(df, n_inputs=len(data_loader.dataset), n_samples_list=n_samples_list,
                                               epsilon=epsilon,
                                               model_idx=idx)

    # filename = "pointwise_softmax_differences_eps=" + str(epsilon_list) \
    #            + "_inputs=" + str(args.inputs) + "_samples="+str(n_samples_list)+"_mode=vi_model=" + str(idx)
    # df = pandas.DataFrame(data={"n_samples": plot_samples, "softmax_differences": plot_softmax_differences,
    #                             "accuracy": plot_original_acc, "epsilon": plot_eps})
    # df.to_pickle(RESULTS+"bnn/"+filename+".pkl")
    # return df


def fgsm_attack(model, image, label, epsilon, device):
    """ Attack a NN model on the given image with an epsilon perturbation.
    :return {"attack","loss_gradient","original_prediction","adversarial_output"}
    """
    image.requires_grad = True
    sum_sign_data_grad = 0.0
    original_output = model.forward(image)
    loss = torch.nn.CrossEntropyLoss()(original_output, label)  # use with softmax

    model.zero_grad()
    loss.backward(retain_graph=True)
    image_grad = image.grad.data
    sum_sign_data_grad = sum_sign_data_grad + image_grad.sign()

    perturbed_image = image + epsilon * sum_sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    adversarial_output = model.forward(perturbed_image)

    return {"perturbed_image":perturbed_image,
            "loss_gradient":image_grad,
            "original_output":original_output,
            "adversarial_output":adversarial_output}

# todo refactor
def fgsm_bayesian_attack(model, n_attack_samples, n_pred_samples, image, label, epsilon, device):
    image.requires_grad = True
    sum_sign_data_grad = 0.0
    original_prediction = model.forward(image, n_samples=n_attack_samples).mean(0)

    loss = torch.nn.CrossEntropyLoss()(original_prediction, label)  # use with softmax

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
    softmax_robustness = softmax_robustness(original_predictions=original_prediction,
                                                 perturbations_predictions=perturbation_prediction)

    if DEBUG:
        print("true_label =", label.item(),
              "\toriginal_pred =", original_prediction,
              "\tperturbation_pred =", perturbation_prediction,
              "\tsoftmax_robustness =", softmax_robustness)

    return {"perturbed_image":perturbed_image, "perturbation_prediction":perturbation_prediction,
            "softmax_robustness":softmax_robustness, "original_prediction":original_prediction}

def attack(model, data_loader, epsilon, device, method="fgsm"):
    """ Attack a NN model on the given inputs with epsilon perturbations.
    :return dictionary {"attacks","loss_gradients","original_accuracy","adversarial_accuracy","softmax_robustness"}
    """

    attacks = []
    loss_gradients = []
    original_outputs = []
    adversarial_outputs =  []

    original_correct = 0.0
    adversarial_correct = 0.0

    for images, labels in data_loader:
        for idx in range(len(images)):
            image = images[idx]
            label = labels[idx]

            input_shape = image.size(0) * image.size(1) * image.size(2)
            label = label.to(device).argmax(-1).view(-1)
            image = image.to(device).view(-1, input_shape)

            attack_dict = fgsm_attack(model=model, image=image, label=label, epsilon=epsilon, device=device)
            # print(attack_dict)

            attacks.append(attack_dict["perturbed_image"])
            loss_gradients.append(attack_dict["loss_gradient"])
            original_outputs.append(attack_dict["original_output"])
            adversarial_outputs.append(attack_dict["adversarial_output"])

            original_correct += ((attack_dict["original_output"].argmax(-1)==label).sum().item())
            adversarial_correct += ((attack_dict["adversarial_output"].argmax(-1)==label).sum().item())

    original_accuracy = 100 * original_correct / len(data_loader.dataset)
    adversarial_accuracy = 100 * adversarial_correct / len(data_loader.dataset)

    softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

    return {"original_accuracy": original_accuracy, "adversarial_accuracy": adversarial_accuracy,
            "softmax_robustness": softmax_rob, "loss_gradients":loss_gradients, "attacks":attacks}

def bayesian_attack(model, n_attack_samples, n_pred_samples, data_loader, epsilon, device, method="fgsm"):
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


    # print(f"acc = {accuracy:.2f}", end="\t")
    print(f"exp_softmax_diff = {exp_softmax_diff:.8f}")
    # print("softmax_rob = ", int(softmax_rob), end="\n")

    return {"accuracy":accuracy,
            "pointwise_softmax_differences":pointwise_softmax_differences,
            "pointwise_softmax_robustness":pointwise_softmax_robustness}#, "min_eps_pert":min_eps_pert}
