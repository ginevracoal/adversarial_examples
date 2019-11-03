# -*- coding: utf-8 -*-

import time
import os
import sys
from random_ensemble import RandomEnsemble
from baseline_convnet import BaselineConvnet
from projection_functions import compute_single_projection
from utils import load_dataset
from joblib import Parallel, delayed

MODEL_NAME = "random_ensemble"
PROJ_MODE = "channels"
RESULTS = "../results/"
BASECLASS = "skl"
DATASETS = "mnist, cifar"


class ParallelRandomEnsemble(RandomEnsemble):

    def __init__(self, input_shape, num_classes, size_proj, proj_idx, data_format, dataset_name, projection_mode):
        super(ParallelRandomEnsemble, self).__init__(input_shape, num_classes, 1, size_proj, projection_mode,
                                                     data_format, dataset_name, test=False)
        self.proj_idx = proj_idx

    def _set_session(self):
        return None

    def train(self, x_train, y_train):
        """ Trains a single projection of the ensemble classifier and saves the model in current day results folder."""

        print("\nTraining single randens projection with seed=", str(self.proj_idx), "and size_proj=", str(self.size_proj))
        start_time = time.time()

        x_train_projected, x_train_inverse_projected = compute_single_projection(input_data=x_train,
                                                                                 seed=self.proj_idx,
                                                                                 size_proj=self.size_proj,
                                                                                 projection_mode=self.projection_mode)

        # eventually adjust input dimension to a single channel projection
        if x_train_projected.shape[3] == 1:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 1)

        # use the same model architecture (not weights) for all trainings
        baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                   dataset_name=self.dataset_name, data_format=self.data_format, test=False)

        classifier = baseline.train(x_train_projected, y_train)
        print("\nProjection + training time: --- %s seconds ---" % (time.time() - start_time))

        self.save_classifier(classifier)

        return classifier

    def save_classifier(self, classifier, model_name=MODEL_NAME):
        start = time.time()
        filename = MODEL_NAME + "_size=" + str(self.size_proj) + "_" + str(self.proj_idx) + ".h5"
        folder = str(self.dataset_name) + "_" + MODEL_NAME + "_size=" + str(self.size_proj) + "_" + str(self.projection_mode) + "/"
        os.makedirs(os.path.dirname(RESULTS + time.strftime('%Y-%m-%d') + "/" + folder), exist_ok=True)
        
        # super(BaselineConvnet, self).save_classifier(classifier=classifier, model_name=folder+filename)

        if BASECLASS == "art":
            classifier.save_classifier(filename=filename, path=RESULTS + time.strftime('%Y-%m-%d') + "/" + folder)
        elif BASECLASS == "skl":
            classifier.model.save_weights(RESULTS + time.strftime('%Y-%m-%d') + "/" + folder + filename)

        saving_time = time.time() - start
        self.training_time -= saving_time


def parallel_train(dataset_name, test, proj_idx, size_proj, proj_mode):
    import tensorflow as tf
    g = tf.Graph()
    with g.as_default():
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)

        model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj,
                                       proj_idx=proj_idx, data_format=data_format, dataset_name=dataset_name,
                                       projection_mode=proj_mode)
        model.train(x_train=x_train, y_train=y_train)
        del tf


def main(dataset_name, test, n_proj, size_proj, proj_mode):
    Parallel(n_jobs=n_proj)(delayed(parallel_train)(dataset_name, test, idx, size_proj, proj_mode) for idx in range(n_proj))

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)
    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj, n_proj=n_proj,
                           data_format=data_format, dataset_name=dataset_name, projection_mode=proj_mode, test=test)

    # ======== buggy =========
    classifiers = model.load_classifier(relative_path=RESULTS + time.strftime('%Y-%m-%d') + "/")
    exit()
    model.evaluate(classifiers, x_test, y_test)
    for attack in ['fgsm','pgd','deepfool','carlini']:
        x_test_adv = model.load_adversaries(dataset_name=dataset_name, attack=attack, eps=0.5, test=test)
        model.evaluate(classifiers, x_test_adv, y_test)


if __name__ == "__main__":

    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        n_proj = int(sys.argv[3])
        size_proj_list = list(map(int, sys.argv[4].strip('[]').split(',')))
        projection_mode = sys.argv[5]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        n_proj = input("\nChoose the number of projections (int): . ")
        size_proj_list = input("\nChoose size for the projection (list): ")
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")

    for size_proj in size_proj_list:
        main(dataset_name=dataset_name, test=test, n_proj=n_proj, size_proj=size_proj, proj_mode=projection_mode)

