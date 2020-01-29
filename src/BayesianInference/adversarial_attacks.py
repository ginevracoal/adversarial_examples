import copy

from utils import *
from robustness_measures import softmax_robustness

DEBUG=False


def fgsm_attack(model, image, label, epsilon, device):
    """ Attack a NN model on the given image with an epsilon perturbation.
    :return {"attack","loss_gradient","original_prediction","adversarial_output"}
    """
    image.requires_grad = True
    original_output = model.forward(image)
    loss = torch.nn.CrossEntropyLoss()(original_output, label)  # use with softmax

    model.zero_grad()
    loss.backward(retain_graph=True)
    image_grad = image.grad.data

    perturbed_image = image + epsilon * image_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    adversarial_output = model.forward(perturbed_image)

    del model

    return {#"perturbed_image":perturbed_image, #"original_image":image,
            "loss_gradient":image_grad, "original_output":original_output, "adversarial_output":adversarial_output}

def fgsm_bayesian_attack(model, n_attack_samples, n_pred_samples, image, label, epsilon, device):
    image.requires_grad = True

    sum_sign_data_grad = 0.0
    for _ in range(n_attack_samples):
        output = model.forward(image, n_samples=1).mean(0)
        loss = torch.nn.CrossEntropyLoss()(output, label)

        model.zero_grad()
        loss.backward(retain_graph=True)
        image_grad = image.grad.data
        # Collect the element-wise sign of the data gradient
        sum_sign_data_grad = sum_sign_data_grad + image_grad.sign()

    original_output = model.forward(image, n_samples=n_pred_samples).mean(0)
    loss = torch.nn.CrossEntropyLoss()(original_output, label)

    model.zero_grad()
    loss.backward(retain_graph=True)
    image_grad = image.grad.data
    # Collect the element-wise sign of the data gradient
    sum_sign_data_grad = sum_sign_data_grad + image_grad.sign()

    perturbed_image = image + epsilon * sum_sign_data_grad / n_attack_samples
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    adversarial_output = model.forward(perturbed_image, n_samples=n_pred_samples).mean(0)

    if DEBUG:
        print("true_label =", label.item(),
              "\toriginal_pred =", original_output,
              "\tperturbation_pred =", adversarial_output,
              "\tsoftmax_robustness =", softmax_robustness)

    del model

    loss_gradient = sum_sign_data_grad/n_attack_samples

    return {#"original_image":image, "perturbed_image": perturbed_image,
            "loss_gradient": loss_gradient,
            "original_output": original_output,"adversarial_output": adversarial_output,
            "n_attack_samples":n_attack_samples,"n_pred_samples":n_pred_samples}


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

            # attack_dict = fgsm_attack(model=copy.deepcopy(model), image=copy.deepcopy(image),
            #                           label=label, epsilon=epsilon, device=device)
            #
            # # attacks.append(attack_dict["perturbed_image"])
            # loss_gradients.append(attack_dict["loss_gradient"])
            # original_outputs.append(attack_dict["original_output"])
            # adversarial_outputs.append(attack_dict["adversarial_output"])

            attack_dict = fgsm_attack(model=copy.deepcopy(model), image=copy.deepcopy(image),
                                      label=label, epsilon=epsilon, device=device)

            original_correct += ((attack_dict["original_output"].argmax(-1)==label).sum().item())
            adversarial_correct += ((attack_dict["adversarial_output"].argmax(-1)==label).sum().item())

    original_accuracy = 100 * original_correct / len(data_loader.dataset)
    adversarial_accuracy = 100 * adversarial_correct / len(data_loader.dataset)

    softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

    return {"original_accuracy": original_accuracy, "adversarial_accuracy": adversarial_accuracy,
            "softmax_robustness": softmax_rob, "loss_gradients":loss_gradients, "epsilon":epsilon}

def bayesian_attack(model, n_attack_samples, n_pred_samples, data_loader, epsilon, device, filename,
                    method="fgsm"):

    print(f"\nFGSM bayesian attack\teps = {epsilon}\tattack_samples = {n_attack_samples}")

    attacks = []
    loss_gradients = []
    original_outputs = []
    adversarial_outputs = []

    original_correct = 0.0
    adversarial_correct = 0.0

    for images, labels in data_loader:
        for idx in range(len(images)):
            image = images[idx]
            label = labels[idx]

            input_shape = image.size(0) * image.size(1) * image.size(2)
            label = label.to(device).argmax(-1).view(-1)
            image = image.to(device).view(-1, input_shape)

            attack_dict = fgsm_bayesian_attack(model=copy.deepcopy(model), n_attack_samples=n_attack_samples,
                                               n_pred_samples=n_pred_samples, image=copy.deepcopy(image),
                                               label=label, epsilon=epsilon, device=device)

            # attacks.append(attack_dict["perturbed_image"])
            loss_gradients.append(attack_dict["loss_gradient"])
            original_outputs.append(attack_dict["original_output"])
            adversarial_outputs.append(attack_dict["adversarial_output"])

            original_correct += ((attack_dict["original_output"].argmax(-1)==label).sum().item())
            adversarial_correct += ((attack_dict["adversarial_output"].argmax(-1)==label).sum().item())

    original_accuracy = 100 * original_correct / len(data_loader.dataset)
    adversarial_accuracy = 100 * adversarial_correct / len(data_loader.dataset)

    print(f"\norig_acc = {original_accuracy}\t\tadv_acc = {adversarial_accuracy}")

    softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

    attack_dict = {"original_accuracy": original_accuracy, "adversarial_accuracy": adversarial_accuracy,
                   "loss_gradients":loss_gradients, "softmax_robustness": softmax_rob, "epsilon":epsilon}

    if filename != None:
        save_to_pickle(relative_path=RESULTS + "bnn/", filename=filename, data=attack_dict)

    return attack_dict
