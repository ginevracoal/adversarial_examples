"""
Methods for evaluating adversarial robustness.
"""

import numpy as np
import torch



def softmax_difference(original_predictions, perturbations_predictions):
    """
    Compute the expected l-inf norm of the difference between predictions and adversarial predictions. This is both a
    global and point-wise robustness measure.
    """
    n_inputs = len(original_predictions)

    if len(original_predictions) != len(perturbations_predictions):
        raise ValueError("\nInput arrays should have the same length.")

    # original_predictions = torch.stack(original_predictions)
    # perturbations_predictions = torch.stack(perturbations_predictions)

    norms = []
    for idx in range(n_inputs):
        # print(original_predictions.shape, original_predictions)
        softmax_diff = original_predictions-perturbations_predictions
        softmax_diff_norm = np.max(np.abs(softmax_diff))#.norm(p=float("inf"))
        norms.append(softmax_diff_norm)

    exp_softmax_diff = np.sum(norms)/n_inputs
    return exp_softmax_diff.squeeze()

def softmax_robustness(original_predictions, perturbations_predictions):
    """
    This method computes the percentage of perturbed points whose change in classification differs from the original one
    by less than 50%, i.e. such that the l-infinite norm of the softmax difference between the original prediction and
    the pertubed one is lower than 0.5. It is a global robustness measure.
    :param classifier: trained classifier
    :param x1: input data
    :param x2: input perturbations
    """
    count = 0
    for i in range(len(original_predictions)):
        if softmax_difference(original_predictions, perturbations_predictions) < 0.5:
            count += 1
    robustness = count / len(original_predictions) * 100
    # print("Softmax robustness: ", robustness)
    return robustness


def min_eps_perturbation(model, n_attack_samples, n_pred_samples, image, label, epsilon, device):
    # todo debug
    from BayesianInference.adversarial_attacks import fgsm_bayesian_attack

    eps = 0.0
    step = 0.1
    # original_prediction = model.forward(image, n_samples=n_attack_samples).mean(0).to(device)
    # perturbation_prediction = original_prediction.to(device)
    attack = fgsm_bayesian_attack(model=model, n_attack_samples=n_attack_samples, n_pred_samples=n_pred_samples,
                                  image=image, label=label, epsilon=epsilon, device=device)
    original_prediction = attack["original_prediction"]
    perturbation_prediction = attack["perturbation_prediction"]

    while original_prediction.argmax(-1) == perturbation_prediction.argmax(-1) and eps<1:
        # print(original_prediction, perturbation_prediction, eps)

        eps += step
        attack = fgsm_bayesian_attack(model=model, n_attack_samples=n_attack_samples, n_pred_samples=n_pred_samples,
                                      image=image, label=label, epsilon=epsilon, device=device)
        original_prediction = attack["original_prediction"]
        perturbation_prediction = attack["perturbation_prediction"]

    if eps<1:
        print(f"min_eps_pert = {eps:.2f}",end="\t")
    else:
        print("min_eps_pert > 1")

    return eps

    # exp_min_pert = np.sum(min_pert) / n_inputs
    # return exp_min_pert.squeeze()


def worst_case_loss():
    raise NotImplementedError

def avg_min_distance():
    raise NotImplementedError

def adversarial_accuracy():
    raise NotImplementedError