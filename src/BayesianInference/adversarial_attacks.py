import sys
sys.path.append(".")
from directories import *

import argparse
from tqdm import tqdm
import pyro
import random
import copy
from BayesianInference.loss_gradients import expected_loss_gradients, expected_loss_gradient, load_loss_gradients, \
    categorical_loss_gradients_norms
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


def pgd_attack(model, image, label, epsilon, device, alpha=2 / 255, iters=5):
    image = image.to(device)
    label = label.to(device)

    original_image = copy.deepcopy(image)
    original_output = model.forward(image)

    for i in range(iters):
        image.requires_grad = True
        output = model.forward(image)
        loss = torch.nn.CrossEntropyLoss()(output, label).to(device)
        model.zero_grad()
        loss.backward()

        perturbed_image = image + alpha * image.grad.sign()

        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        image = torch.clamp(original_image + eta, min=0, max=1).detach_()

    adversarial_output = model.forward(perturbed_image)
    return {"original_output":original_output, "adversarial_output":adversarial_output}


def random_bayesian_attack(model, image, label, n_pred_samples, epsilon, device):

    original_output = model.forward(image, n_samples=n_pred_samples).mean(0).to(device)
    random_pert = torch.tensor([random.choice([-1,0,1]) for _ in range(784)]).to(device)
    perturbed_image = image + epsilon * random_pert
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    adversarial_output = model.forward(perturbed_image, n_samples=n_pred_samples).mean(0)

    if DEBUG:
        print("true_label =", label.item(), "\tperturbation_pred =", adversarial_output)

    del model

    return {"original_output": original_output, "adversarial_output": adversarial_output,
            "n_pred_samples": n_pred_samples}

def fgsm_bayesian_attack(model, n_attack_samples, n_pred_samples, image, label, epsilon, device, loss_gradient=None):

    original_output = model.forward(image, n_samples=n_pred_samples).mean(0).to(device)

    if loss_gradient is None:
        loss_gradient = expected_loss_gradient(posterior=model, n_samples=n_attack_samples,
                                                    image=image, label=label, device=device, mode="vi")
    perturbed_image = image + epsilon * loss_gradient.to(device)

    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    adversarial_output = model.forward(perturbed_image, n_samples=n_pred_samples).mean(0)

    if DEBUG:
        print("true_label =", label.item(),
              "\tperturbation_pred =", adversarial_output,
              "\tsoftmax_robustness =", softmax_robustness)

    del model

    attack_dict = {"original_output": original_output,"adversarial_output": adversarial_output,
            "n_attack_samples":n_attack_samples,"n_pred_samples":n_pred_samples}
    return attack_dict

def pgd_bayesian_attack(model, n_attack_samples, n_pred_samples, image, label, epsilon, device, alpha=2 / 255, iters=10):
    image = image.to(device)
    label = label.to(device)

    original_image = copy.deepcopy(image)
    original_output = model.forward(image, n_samples=n_pred_samples).mean(0).to(device)

    for i in range(iters):
        image.requires_grad = True
        output = model.forward(image, n_samples=n_attack_samples).mean(0).to(device)
        loss = torch.nn.CrossEntropyLoss()(output, label).to(device)
        model.zero_grad()
        loss.backward()

        perturbed_image = image + alpha * image.grad.sign()

        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        image = torch.clamp(original_image + eta, min=0, max=1).detach_()

    adversarial_output = model.forward(perturbed_image, n_samples=n_pred_samples).mean(0).to(device)

    attack_dict = {"original_output": original_output,"adversarial_output": adversarial_output,
                   "n_attack_samples":n_attack_samples,"n_pred_samples":n_pred_samples}
    return attack_dict


def attack(model, data_loader, dataset_name, epsilon, device, method="fgsm", model_idx=0, save_dir=RESULTS+"attacks/"):

    original_outputs = []
    adversarial_outputs = []

    original_correct = 0.0
    adversarial_correct = 0.0

    for images, labels in tqdm(data_loader):
        for idx in range(len(images)):
            image = images[idx]
            label = labels[idx]

            input_shape = image.size(0) * image.size(1) * image.size(2)
            label = label.to(device).argmax(-1).view(-1)
            image = image.to(device).view(-1, input_shape)

            if method == "fgsm":
                attack_dict = fgsm_attack(model=copy.deepcopy(model), image=copy.deepcopy(image),
                                      label=label, epsilon=epsilon, device=device)
            elif method == "pgd":
                attack_dict = pgd_attack(model=copy.deepcopy(model), image=copy.deepcopy(image),
                                          label=label, epsilon=epsilon, device=device)

            original_correct += ((attack_dict["original_output"].argmax(-1) == label).sum().item())
            adversarial_correct += ((attack_dict["adversarial_output"].argmax(-1) == label).sum().item())

            original_outputs.append(attack_dict["original_output"])
            adversarial_outputs.append(attack_dict["adversarial_output"])

    original_accuracy = 100 * original_correct / len(data_loader.dataset)
    adversarial_accuracy = 100 * adversarial_correct / len(data_loader.dataset)
    print(f"\norig_acc = {original_accuracy}\t\tadv_acc = {adversarial_accuracy}")
    softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

    filename = str(dataset_name)+"_nn_inputs="+str(len(data_loader.dataset))+"_eps="+str(epsilon)\
               +"_attack_"+str(model_idx)+".pkl"
    dict = {"original_accuracy": original_accuracy, "adversarial_accuracy": adversarial_accuracy,
            "softmax_robustness": softmax_rob}
    save_to_pickle(data=dict, relative_path=save_dir, filename=filename)
    return dict


def bayesian_attack(model, n_attack_samples, data_loader, dataset_name, epsilon, device, model_idx,
                    loss_gradients=None, n_pred_samples=None, save_dir=RESULTS+"attacks/", method="fgsm"):

    if n_pred_samples is None:
        n_pred_samples = n_attack_samples

    if loss_gradients is None:
        loss_gradients = expected_loss_gradients(posterior=model, n_samples=n_attack_samples, dataset_name=dataset_name,
                                                 model_idx=model_idx, data_loader=data_loader, device=device, mode="vi")

    original_outputs = []
    adversarial_outputs = []

    original_correct = 0.0
    adversarial_correct = 0.0

    images_count = 0
    for images, labels in data_loader:
        for idx in tqdm(range(len(images))):
            image = images[idx]
            label = labels[idx]
            # input_shape = image.size(0) * image.size(1) * image.size(2)
            label = label.to(device).argmax(-1).view(-1)
            image = image.to(device).flatten()#view(-1, input_shape)
            loss_gradient = torch.tensor(loss_gradients[images_count])
            # print(image.shape, loss_gradient.shape)

            if method == "fgsm":
                attack_dict = fgsm_bayesian_attack(model=copy.deepcopy(model), n_attack_samples=n_attack_samples,
                                                   n_pred_samples=n_pred_samples, image=copy.deepcopy(image),
                                                   label=label, epsilon=epsilon, device=device,
                                                   loss_gradient=loss_gradient)
            elif method == "pgd":
                attack_dict = pgd_bayesian_attack(model=copy.deepcopy(model), n_attack_samples=n_attack_samples,
                                                   n_pred_samples=n_pred_samples, image=copy.deepcopy(image),
                                                   label=label, epsilon=epsilon, device=device)

            original_outputs.append(attack_dict["original_output"])
            adversarial_outputs.append(attack_dict["adversarial_output"])
            original_correct += ((attack_dict["original_output"].argmax(-1)==label).sum().item())
            adversarial_correct += ((attack_dict["adversarial_output"].argmax(-1)==label).sum().item())
            images_count += 1

    original_accuracy = 100 * original_correct / len(data_loader.dataset)
    adversarial_accuracy = 100 * adversarial_correct / len(data_loader.dataset)
    print(f"\norig_acc = {original_accuracy} - adv_acc = {adversarial_accuracy} - ", end="")
    softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

    filename = str(dataset_name)+"_bnn_inputs="+str(len(data_loader.dataset))+"_attackSamp="+str(n_attack_samples)+\
               "_predSamp="+str(n_pred_samples)+"_eps="+str(epsilon)+"_attack_"+str(model_idx)+".pkl"
    dict = {"original_accuracy": original_accuracy, "adversarial_accuracy": adversarial_accuracy,
            "softmax_robustness": softmax_rob}
    save_to_pickle(data=dict, relative_path=save_dir, filename=filename)

    return dict


def categorical_bayesian_attack(model, n_attack_samples, data_loader, dataset_name, epsilon, device, model_idx,
                                loss_gradients, loss_gradients_categories, n_pred_samples=None, method="fgsm"):

    cat_attack_dict = {"epsilon":epsilon, "n_samples":n_attack_samples}

    if n_pred_samples is None:
        n_pred_samples = n_attack_samples

    for category in ["vanishing","const_null","other"]:
        print(f"\nCategory = {category}\n")
        losses = []
        original_outputs = []
        adversarial_outputs = []

        original_correct = 0.0
        adversarial_correct = 0.0

        image_idx_count = 0
        cat_count = 0
        for images, labels in tqdm(data_loader):
            for idx in range(len(images)):
                if loss_gradients_categories[image_idx_count] == category:
                    cat_count += 1
                    image = images[idx]
                    label = labels[idx]
                    # input_shape = image.size(0) * image.size(1) * image.size(2)
                    label = label.to(device).argmax(-1).view(-1)
                    image = image.to(device).flatten()#view(-1, input_shape)
                    # exit()
                    loss_gradient = torch.tensor(loss_gradients[image_idx_count])
                    # print(image.shape, loss_gradient.shape)
                    attack_dict = fgsm_bayesian_attack(model=copy.deepcopy(model), n_attack_samples=n_attack_samples,
                                                       n_pred_samples=n_pred_samples, image=copy.deepcopy(image),
                                                       label=label, epsilon=epsilon, device=device,
                                                       loss_gradient=loss_gradient)

                    original_outputs.append(attack_dict["original_output"])
                    adversarial_outputs.append(attack_dict["adversarial_output"])
                    losses.append(attack_dict["loss"].item())
                    original_correct += ((attack_dict["original_output"].argmax(-1)==label).sum().item())
                    adversarial_correct += ((attack_dict["adversarial_output"].argmax(-1)==label).sum().item())
                image_idx_count += 1

        original_accuracy = 100 * original_correct / cat_count
        adversarial_accuracy = 100 * adversarial_correct / cat_count
        print(f"\norig_acc = {original_accuracy} - adv_acc = {adversarial_accuracy} - ", end="")
        softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

        category_dict = {"original_accuracy": original_accuracy, "adversarial_accuracy": adversarial_accuracy,
                         "softmax_robustness": softmax_rob, "losses":losses, "n_inputs":len(losses)}
        cat_attack_dict.update({category:category_dict})

    print("cat_attack_dict.keys() :", cat_attack_dict.keys())

    filename = str(dataset_name)+"_bnn_inputs="+str(len(data_loader.dataset))\
               +"_samples="+str(n_attack_samples)+"_eps="+str(epsilon)+"_categorical_attack"+str(model_idx)+".pkl"
    save_to_pickle(data=cat_attack_dict, relative_path=RESULTS+"attacks/", filename=filename)
    return cat_attack_dict


# def bnn_create_save_data_pretrained_models(dataset_name, device):
#     train_loader, test_loader, data_format, input_shape = \
#         data_loaders(dataset_name=dataset_name, batch_size=64, n_inputs=60000, shuffle=True)
#
#
#     accuracy = []
#     robustness = []
#     epsilon = []
#     model_type = []
#
#     # === load models ===
#     count = 82
#
#     for idx in [0]:
#
#         for test_slice in [500]:
#             test = slice_data_loader(test_loader, slice_size=test_slice)
#             model_dict = models_list[idx]
#
#             bayesnn = VI_BNN(input_shape=input_shape, device=device, architecture=model_dict["activation"],
#                              activation=model_dict["activation"])
#             posterior = bayesnn.load_posterior(posterior_name=model_dict["filename"], relative_path=TRAINED_MODELS,
#                                                    activation=model_dict["activation"])
#             for eps in [0.1, 0.3, 0.6]:
#                 for n_samples in [2]:
#                     posterior.evaluate(data_loader=test, n_samples=n_samples)
#
#                     filename = str(model_dict["dataset"])+"_eps="+str(eps)+"_samples="+str(n_samples)+"_attacks.pkl"
#                     attack_dict = bayesian_attack(model=posterior, data_loader=test, epsilon=eps, device=device,
#                                                   n_attack_samples=n_samples, n_pred_samples=n_samples,
#                                                   dataset_name=dataset_name)
#
#                     robustness.append(attack_dict["softmax_robustness"])
#                     accuracy.append(attack_dict["original_accuracy"])
#                     epsilon.append(eps)
#                     model_type.append("bnn")
#
#                     count += 1
#                     filename = str(dataset_name) + "_bnn_attack_" + str(count) + ".pkl"
#                     save_to_pickle(relative_path=RESULTS + "bnn/", filename=filename, data=attack_dict)
#
#     scatterplot_accuracy_robustness(accuracy=accuracy, robustness=robustness, model_type=model_type, epsilon=epsilon)
#     return robustness, accuracy, epsilon, model_type


# def create_save_data(dataset_name, device):
#
#     model_type = []
#     accuracy = []
#     robustness = []
#     epsilon = []
#
#
#     nn_robustness, nn_accuracy, nn_epsilon, nn_model_type = nn_create_save_data(dataset_name, device)
#     model_type.extend(nn_model_type)
#     robustness.extend(nn_robustness)
#     accuracy.extend(nn_accuracy)
#     epsilon.extend(nn_epsilon)
#
#     # bnn_robustness, bnn_accuracy, bnn_epsilon, bnn_model_type = bnn_create_save_data_pretrained_models(dataset_name, device)
#     bnn_robustness, bnn_accuracy, bnn_epsilon, bnn_model_type = bnn_create_save_data(dataset_name, device)
#     model_type.extend(bnn_model_type)
#     robustness.extend(bnn_robustness)
#     accuracy.extend(bnn_accuracy)
#     epsilon.extend(bnn_epsilon)
#
#     return robustness, accuracy, epsilon, model_type


def main(args):

    # = initialization =
    n_inputs, n_samples_list, model_idx, dataset = 1000, [1,5,10,50,100], 2, "mnist"

    train_loader, test_loader, data_format, input_shape = \
        data_loaders(dataset_name=dataset, batch_size=128, n_inputs=n_inputs, shuffle=True)
    # test_loader = slice_data_loader(test_loader, slice_size=100)

    # = load the posterior =
    model = hidden_vi_models[model_idx]
    bayesnn = VI_BNN(input_shape=input_shape, device=args.device, architecture=model["architecture"],
                     activation=model["activation"])
    posterior = bayesnn.load_posterior(posterior_name=model["filename"], relative_path=TRAINED_MODELS,
                                       activation=model["activation"])

    # = attack the posterior =
    # for n_samples in n_samples_list:
    #     loss_gradients = load_loss_gradients(dataset_name=dataset, n_inputs=n_inputs, n_samples=n_samples,
    #                                          model_idx=model_idx)
    #
    #     bayesian_attack(model=posterior, n_attack_samples=n_samples, data_loader=test_loader,
    #                     dataset_name=dataset, epsilon=0.25, device=args.device, model_idx=model_idx,
    #                     loss_gradients=loss_gradients, n_pred_samples=None, method="fgsm")

    # = categorical attack =

    filename = str(model["dataset"])+"_bnn_inputs="+str(len(test_loader.dataset))+\
               "_samples="+str(n_samples_list)+"_cat_lossGrads_norms"+str(model_idx)+".pkl"
    loss_gradients_categories = load_from_pickle(path=RESULTS+filename)

    for n_samples in n_samples_list:
        loss_gradients = load_loss_gradients(dataset_name=dataset, n_inputs=n_inputs, n_samples=n_samples,
                                             model_idx=model_idx)

        categorical_bayesian_attack(model=posterior, n_attack_samples=n_samples, data_loader=test_loader,
                                    dataset_name=model["dataset"], epsilon=0.25, device=args.device,
                                    model_idx=model_idx, loss_gradients=loss_gradients,
                                    loss_gradients_categories=loss_gradients_categories)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.1.0')
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs='?', default="mnist", type=str)
    parser.add_argument("--device", default='cpu', type=str, help='use "cpu" or "cuda".')

    main(args=parser.parse_args())
