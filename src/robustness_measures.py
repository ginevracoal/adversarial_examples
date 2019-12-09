"""
Methods for evaluating adversarial robustness.
"""

import numpy as np


def softmax_difference(classifier, x1, x2):
    """
    Compute the expected l-infinite norm of the difference between prediction and adversarial prediction.
    :param classifier: trained classifier
    :param x1: input data
    :param x2: input perturbations
    """

    if len(x1) != len(x2):
        raise ValueError("\nInput arrays should have the same length.")

    if x1.ndim==3:
        x1 = np.expand_dims(x1, axis=0)
        x2 = np.expand_dims(x2, axis=0)

    x1_predictions = classifier.predict(x=x1)
    x2_predictions = classifier.predict(x=x2)

    diff = np.diff([x1_predictions,x2_predictions],axis=0)[0]
    # print("\ncheck:\n",x1_predictions[0],"\n-",x2_predictions[0],"\n=",diff[0,:])
    norm = np.linalg.norm(diff, axis=1, ord=np.inf)
    exp = np.sum(norm)
    if x1.shape[0]!=1:
        print("Expected softmax difference: ", exp/x1.shape[0])
    return exp

def softmax_robustness(classifier, x1, x2):
    """
    This method computes the percentage of perturbed points whose change in classification differs from the original one
    by less than 50%, i.e. such that the l-infinite norm of the softmax difference between the original prediction and
    the pertubed one is lower than 0.5
    :param classifier: trained classifier
    :param x1: input data
    :param x2: input perturbations
    """
    if len(x1) != len(x2):
        raise ValueError("\nInput arrays should have the same length.")
    count = 0
    for i in range(len(x1)):
        if softmax_difference(classifier=classifier, x1=x1[i], x2=x2[i]) < 0.5:
            count += 1
    robustness = count / len(x1)
    print("Softmax robustness: ", robustness)
    return robustness

def worst_case_loss():
    raise NotImplementedError

def avg_min_distance():
    raise NotImplementedError

def adversarial_accuracy():
    raise NotImplementedError