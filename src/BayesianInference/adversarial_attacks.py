import torch
import torch.nn.functional as F
from torch.distributions import Uniform


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


def fgsm_bayesian_attack(model, n_samples, image, label, epsilon):
    image.requires_grad = True
    sum_sign_data_grad = 0.0
    for _ in range(n_samples):
        sampled_model = model.guide(None, None)
        output = sampled_model(image)

        loss = F.cross_entropy(output, label)
        # zero gradients
        sampled_model.zero_grad()
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

        perturbed_image = fgsm_bayesian_attack(model, n_samples, image, label, epsilon)
        prediction = model.forward(perturbed_image)
        if prediction.item() == label.item():
            correct += 1
        adv_examples.append((prediction.item(), prediction.item(), perturbed_image.squeeze().detach().to(device).numpy()))

    accuracy = correct / float(len(data_loader))
    print("\nAttack epsilon = {}\t Accuracy = {} / {} = {}".format(epsilon, correct, len(data_loader), accuracy))
    return accuracy, adv_examples


def expected_loss_gradient(model, n_samples, image, label, mode):
    image.requires_grad = True
    loss_gradient = 0.0
    for _ in range(n_samples):
        if mode == "hidden":
            raw_output = model.forward(image, n_samples=1).mean(0)
            loss = F.cross_entropy(raw_output, label)
            loss.backward(retain_graph=True)
            loss_gradient = loss_gradient + image.grad.data
        else:
            sampled_model = model.guide(None)
            output = sampled_model(image)
            # todo:outputs should be the raw unnormalized outputs, but they're one hot
            print(output)
            exit()
            loss = F.cross_entropy(output, label)
            # zero gradients
            sampled_model.zero_grad()
            # compute gradients
            loss.backward(retain_graph=True)
            # Create the perturbed image by adjusting each pixel of the input image
            # print(image.grad.data[0][0:10])
            loss_gradient = loss_gradient + image.grad.data

    exp_loss_gradient = loss_gradient / n_samples
    print(f"mean = {loss_gradient[0].mean().item():.3f} \t var = {loss_gradient[0].var().item():.3f}")
    return exp_loss_gradient


def expected_loss_gradients(model, n_samples, data_loader, device="cpu", mode=None):
    print("\nExpected loss gradients:")
    expected_loss_gradients = []

    for image, label in data_loader:
        if image.size(0) != 1:
            raise ValueError("data_loader batch_size should be 1.")

        input_shape = image.size(1) * image.size(2) * image.size(3)
        label = label.to(device).argmax(-1)
        image = image.to(device).view(-1, input_shape)
        expected_loss_gradients.append(expected_loss_gradient(model, n_samples, image, label, mode=mode))

    return expected_loss_gradients