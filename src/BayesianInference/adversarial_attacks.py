import sys
sys.path.append(".")
from directories import *

import argparse
from tqdm import tqdm
import pyro
import copy
from BayesianInference.loss_gradients import expected_loss_gradients, expected_loss_gradient, load_loss_gradients
from BayesianInference.pyro_utils import data_loaders, slice_data_loader
from utils import *
from robustness_measures import softmax_robustness
from BayesianInference.hidden_vi_bnn import hidden_vi_models
from BayesianInference.hidden_vi_bnn import VI_BNN

DEBUG=False
DATA_PATH="../data/attacks/"


def fgsm_attack(model, image, label, epsilon, device):
    """ Attack a NN model on the given image with an epsilon perturbation.
    :return {"attack","loss_gradient","original_prediction","adversarial_output"}
    """
    image.requires_grad = True
    original_output = model.forward(image)
    loss = torch.nn.CrossEntropyLoss()(original_output, label)

    model.zero_grad()
    loss.backward(retain_graph=True)
    image_grad = image.grad.data

    perturbed_image = image + epsilon * image_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    adversarial_output = model.forward(perturbed_image)

    del model

    return {"loss_gradient":image_grad, "original_output":original_output, "adversarial_output":adversarial_output}


def fgsm_bayesian_attack(model, n_attack_samples, n_pred_samples, image, label, epsilon, device, loss_gradient=None):
    image.requires_grad = True

    ### old ###
    # sum_sign_data_grad = 0.0
    # for _ in range(n_attack_samples):
    #     output = model.forward(image, n_samples=1).mean(0)
    #     loss = torch.nn.CrossEntropyLoss()(output, label)
    #
    #     model.zero_grad()
    #     loss.backward(retain_graph=True)
    #     image_grad = image.grad.data
    #     # Collect the element-wise sign of the data gradient
    #     sum_sign_data_grad = sum_sign_data_grad + image_grad.sign()
    #
    # perturbed_image = image + epsilon * sum_sign_data_grad / n_attack_samples
    # loss_gradient = sum_sign_data_grad/n_attack_samples

    original_output = model.forward(image, n_samples=1).mean(0)
    loss = torch.nn.CrossEntropyLoss()(original_output, label)

    ### new ###
    if loss_gradient is None:
        loss_gradient = expected_loss_gradient(posterior=model, n_samples=n_attack_samples,
                                                    image=image, label=label, device=device, mode="vi")
    perturbed_image = image + epsilon * loss_gradient

    ###########

    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    adversarial_output = model.forward(perturbed_image, n_samples=n_pred_samples).mean(0)

    if DEBUG:
        print("true_label =", label.item(),
              "\tperturbation_pred =", adversarial_output,
              "\tsoftmax_robustness =", softmax_robustness)

    del model

    return {"original_output": original_output,"adversarial_output": adversarial_output, "loss":loss,
            "n_attack_samples":n_attack_samples,"n_pred_samples":n_pred_samples}


# def old_attack(model, data_loader, epsilon, device, loss_gradients=None, method="fgsm"):
#     """ Attack a NN model on the given inputs with epsilon perturbations.
#     :return dictionary {"attacks","loss_gradients","original_accuracy","adversarial_accuracy","softmax_robustness"}
#     """
#
#     attacks = []
#     loss_gradients = []
#     original_outputs = []
#     adversarial_outputs =  []
#
#     original_correct = 0.0
#     adversarial_correct = 0.0
#
#     for images, labels in data_loader:
#         for idx in range(len(images)):
#             image = images[idx]
#             label = labels[idx]
#
#             input_shape = image.size(0) * image.size(1) * image.size(2)
#             label = label.to(device).argmax(-1).view(-1)
#             image = image.to(device).view(-1, input_shape)
#
#             # attack_dict = fgsm_attack(model=copy.deepcopy(model), image=copy.deepcopy(image),
#             #                           label=label, epsilon=epsilon, device=device)
#             #
#             # # attacks.append(attack_dict["perturbed_image"])
#             # loss_gradients.append(attack_dict["loss_gradient"])
#             # original_outputs.append(attack_dict["original_output"])
#             # adversarial_outputs.append(attack_dict["adversarial_output"])
#
#             attack_dict = fgsm_attack(model=copy.deepcopy(model), image=copy.deepcopy(image),
#                                       label=label, epsilon=epsilon, device=device)
#
#             original_correct += ((attack_dict["original_output"].argmax(-1)==label).sum().item())
#             adversarial_correct += ((attack_dict["adversarial_output"].argmax(-1)==label).sum().item())
#
#     original_accuracy = 100 * original_correct / len(data_loader.dataset)
#     adversarial_accuracy = 100 * adversarial_correct / len(data_loader.dataset)
#
#     softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)
#
#     return {"original_accuracy": original_accuracy, "adversarial_accuracy": adversarial_accuracy,
#             "softmax_robustness": softmax_rob, "loss_gradients":loss_gradients, "epsilon":epsilon}


def attack(model, data_loader, dataset_name, epsilon, device, method="fgsm"):

    original_outputs = []
    adversarial_outputs = []

    original_correct = 0.0
    adversarial_correct = 0.0

    for images, labels in data_loader:
        for idx in tqdm(range(len(images))):
            image = images[idx]
            label = labels[idx]

            input_shape = image.size(0) * image.size(1) * image.size(2)
            label = label.to(device).argmax(-1).view(-1)
            image = image.to(device).view(-1, input_shape)

            attack_dict = fgsm_attack(model=copy.deepcopy(model), image=copy.deepcopy(image),
                                      label=label, epsilon=epsilon, device=device)

            original_correct += ((attack_dict["original_output"].argmax(-1) == label).sum().item())
            adversarial_correct += ((attack_dict["adversarial_output"].argmax(-1) == label).sum().item())

    original_accuracy = 100 * original_correct / len(data_loader.dataset)
    adversarial_accuracy = 100 * adversarial_correct / len(data_loader.dataset)

    softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

    filename = str(dataset_name)+"_nn_inputs="+str(len(data_loader.dataset))+"_attack.pkl"
    dict = {"original_accuracy": original_accuracy, "adversarial_accuracy": adversarial_accuracy,
            "softmax_robustness": softmax_rob, "epsilon":epsilon}
    save_to_pickle(data=dict, relative_path=RESULTS+"attacks/", filename=filename)
    return dict


# def old_bayesian_attack(model, n_attack_samples, n_pred_samples, data_loader, epsilon, device, filename,
#                     method="fgsm"):
#
#     print(f"\nFGSM bayesian attack\teps = {epsilon}\tattack_samples = {n_attack_samples}")
#
#     attacks = []
#     loss_gradients = []
#     original_outputs = []
#     adversarial_outputs = []
#
#     original_correct = 0.0
#     adversarial_correct = 0.0
#
#     for images, labels in data_loader:
#         for idx in range(len(images)):
#             image = images[idx]
#             label = labels[idx]
#
#             input_shape = image.size(0) * image.size(1) * image.size(2)
#             label = label.to(device).argmax(-1).view(-1)
#             image = image.to(device).view(-1, input_shape)
#
#             attack_dict = fgsm_bayesian_attack(model=copy.deepcopy(model), n_attack_samples=n_attack_samples,
#                                                n_pred_samples=n_pred_samples, image=copy.deepcopy(image),
#                                                label=label, epsilon=epsilon, device=device)
#
#             # attacks.append(attack_dict["perturbed_image"])
#             loss_gradients.append(attack_dict["loss_gradient"])
#             original_outputs.append(attack_dict["original_output"])
#             adversarial_outputs.append(attack_dict["adversarial_output"])
#             original_correct += ((attack_dict["original_output"].argmax(-1)==label).sum().item())
#             adversarial_correct += ((attack_dict["adversarial_output"].argmax(-1)==label).sum().item())
#
#     original_accuracy = 100 * original_correct / len(data_loader.dataset)
#     adversarial_accuracy = 100 * adversarial_correct / len(data_loader.dataset)
#     print(f"\norig_acc = {original_accuracy}\t\tadv_acc = {adversarial_accuracy}")
#     softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)
#
#
#     attack_dict = {"original_accuracy": original_accuracy, "adversarial_accuracy": adversarial_accuracy,
#                    "loss_gradients":loss_gradients, "softmax_robustness": softmax_rob, "epsilon":epsilon}
#     if filename != None:
#         save_to_pickle(relative_path=RESULTS + "bnn/", filename=filename, data=attack_dict)
#     return attack_dict


def bayesian_attack(model, n_attack_samples, data_loader, dataset_name, epsilon, device, model_idx,
                    loss_gradients=None, n_pred_samples=None, method="fgsm"):

    if n_pred_samples is None:
        n_pred_samples = n_attack_samples

    if loss_gradients is None:
        loss_gradients = expected_loss_gradients(posterior=model, n_samples=n_attack_samples, dataset_name=dataset_name,
                                                 model_idx=model_idx, data_loader=data_loader, device=device, mode="vi")

    losses = []
    original_outputs = []
    adversarial_outputs = []

    original_correct = 0.0
    adversarial_correct = 0.0

    count = 0
    for images, labels in data_loader:
        for idx in tqdm(range(len(images))):
            image = images[idx]
            label = labels[idx]
            # input_shape = image.size(0) * image.size(1) * image.size(2)
            label = label.to(device).argmax(-1).view(-1)
            image = image.to(device).flatten()#view(-1, input_shape)
            # exit()
            loss_gradient = torch.tensor(loss_gradients[count])
            # print(image.shape, loss_gradient.shape)
            attack_dict = fgsm_bayesian_attack(model=copy.deepcopy(model), n_attack_samples=n_attack_samples,
                                               n_pred_samples=n_pred_samples, image=copy.deepcopy(image),
                                               label=label, epsilon=epsilon, device=device,
                                               loss_gradient=loss_gradient)

            original_outputs.append(attack_dict["original_output"])
            adversarial_outputs.append(attack_dict["adversarial_output"])
            losses.append(attack_dict["loss"])
            original_correct += ((attack_dict["original_output"].argmax(-1)==label).sum().item())
            adversarial_correct += ((attack_dict["adversarial_output"].argmax(-1)==label).sum().item())
            count += 1

    original_accuracy = 100 * original_correct / len(data_loader.dataset)
    adversarial_accuracy = 100 * adversarial_correct / len(data_loader.dataset)
    print(f"\norig_acc = {original_accuracy}\t\tadv_acc = {adversarial_accuracy}")
    softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

    # todo test n_samples here
    filename = str(dataset_name)+"_bnn_inputs="+str(len(data_loader.dataset))+"_samples="+str(1)+"_losses.pkl"
    save_to_pickle(data=losses, relative_path=RESULTS+"losses/", filename=filename)

    filename = str(dataset_name)+"_bnn_inputs="+str(len(data_loader.dataset))\
               +"_samples="+str(n_attack_samples)+"_eps="+str(epsilon)+"_attack.pkl"
    dict = {"original_accuracy": original_accuracy, "adversarial_accuracy": adversarial_accuracy,
            "softmax_robustness": softmax_rob, "epsilon":epsilon, "n_samples":n_attack_samples}
    save_to_pickle(data=dict, relative_path=RESULTS+"attacks/", filename=filename)

    return dict


def bnn_create_save_data_pretrained_models(dataset_name, device):
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset_name, batch_size=64, n_inputs=60000, shuffle=True)


    accuracy = []
    robustness = []
    epsilon = []
    model_type = []

    # === load models ===
    count = 82

    for idx in [0]:

        for test_slice in [500]:
            test = slice_data_loader(test_loader, slice_size=test_slice)
            model_dict = models_list[idx]

            bayesnn = VI_BNN(input_shape=input_shape, device=device, architecture=model_dict["activation"],
                             activation=model_dict["activation"])
            posterior = bayesnn.load_posterior(posterior_name=model_dict["filename"], relative_path=TRAINED_MODELS,
                                                   activation=model_dict["activation"])
            for eps in [0.1, 0.3, 0.6]:
                for n_samples in [2]:
                    posterior.evaluate(data_loader=test, n_samples=n_samples)

                    filename = str(model_dict["dataset"])+"_eps="+str(eps)+"_samples="+str(n_samples)+"_attacks.pkl"
                    attack_dict = bayesian_attack(model=posterior, data_loader=test, epsilon=eps, device=device,
                                                  n_attack_samples=n_samples, n_pred_samples=n_samples,
                                                  dataset_name=dataset_name)

                    robustness.append(attack_dict["softmax_robustness"])
                    accuracy.append(attack_dict["original_accuracy"])
                    epsilon.append(eps)
                    model_type.append("bnn")

                    count += 1
                    filename = str(dataset_name) + "_bnn_attack_" + str(count) + ".pkl"
                    save_to_pickle(relative_path=RESULTS + "bnn/", filename=filename, data=attack_dict)

    scatterplot_accuracy_robustness(accuracy=accuracy, robustness=robustness, model_type=model_type, epsilon=epsilon)
    return robustness, accuracy, epsilon, model_type


def create_save_data(dataset_name, device):

    model_type = []
    accuracy = []
    robustness = []
    epsilon = []


    nn_robustness, nn_accuracy, nn_epsilon, nn_model_type = nn_create_save_data(dataset_name, device)
    model_type.extend(nn_model_type)
    robustness.extend(nn_robustness)
    accuracy.extend(nn_accuracy)
    epsilon.extend(nn_epsilon)

    # bnn_robustness, bnn_accuracy, bnn_epsilon, bnn_model_type = bnn_create_save_data_pretrained_models(dataset_name, device)
    bnn_robustness, bnn_accuracy, bnn_epsilon, bnn_model_type = bnn_create_save_data(dataset_name, device)
    model_type.extend(bnn_model_type)
    robustness.extend(bnn_robustness)
    accuracy.extend(bnn_accuracy)
    epsilon.extend(bnn_epsilon)

    return robustness, accuracy, epsilon, model_type


def main(args):

    # n_inputs, n_samples_list, model_idx, dataset = 1000, [1,5,10,50,100], 2, "mnist"
    n_inputs, n_samples_list, model_idx, dataset = 1000, [1,5,10,50,100], 2, "mnist"
    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset, batch_size=128, n_inputs=n_inputs, shuffle=True)
    test_loader = slice_data_loader(test_loader, slice_size=100)

    model = hidden_vi_models[model_idx]
    bayesnn = VI_BNN(input_shape=input_shape, device=args.device, architecture=model["architecture"],
                     activation=model["activation"])
    posterior = bayesnn.load_posterior(posterior_name=model["filename"], relative_path=TRAINED_MODELS,
                                       activation=model["activation"])

    for n_samples in n_samples_list:
        loss_gradients = load_loss_gradients(dataset_name=dataset, n_inputs=n_inputs, n_samples=n_samples,
                                             model_idx=model_idx)

        bayesian_attack(model=posterior, n_attack_samples=n_samples, data_loader=test_loader,
                        dataset_name=dataset, epsilon=0.25, device=args.device, model_idx=model_idx,
                        loss_gradients=loss_gradients, n_pred_samples=None, method="fgsm")


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')

    main(args=parser.parse_args())
