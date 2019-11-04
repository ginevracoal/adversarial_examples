# -*- coding: utf-8 -*-

from random_ensemble import *
from projection_functions import compute_single_projection
from utils import load_dataset
from joblib import Parallel, delayed

MODEL_NAME = "random_ensemble"
PROJ_MODE = "channels"
DATASETS = "mnist, cifar"


class ParallelRandomEnsemble(RandomEnsemble):

    def __init__(self, input_shape, num_classes, size_proj, n_proj, data_format, dataset_name, projection_mode, test):
        super(ParallelRandomEnsemble, self).__init__(input_shape, num_classes, n_proj, size_proj, projection_mode,
                                                     data_format, dataset_name, test=test)

    def _set_session(self):
        return None

    def train_single_projection(self, x_train, y_train, device, proj_idx):
        """ Trains a single projection of the ensemble classifier and saves the model in current day results folder."""

        print("\nTraining single randens projection with seed=", str(proj_idx),
              "and size_proj=", str(self.size_proj))

        start_time = time.time()
        x_train_projected, x_train_inverse_projected = compute_single_projection(input_data=x_train,
                                                                                 seed=proj_idx,
                                                                                 size_proj=self.size_proj,
                                                                                 projection_mode=self.projection_mode)

        # eventually adjust input dimension to a single channel projection
        if x_train_projected.shape[3] == 1:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 1)

        # use the same model architecture (not weights) for all trainings
        randens = RandomEnsemble(input_shape=self.input_shape, num_classes=self.num_classes, n_proj=self.n_proj,
                                  size_proj=self.size_proj, projection_mode=self.projection_mode,
                                   dataset_name=self.dataset_name, data_format=self.data_format, test=False)

        # baseline.model_name = self.dataset_name + "_size=" + str(self.size_proj) + "_" + str(self.proj_idx) + self.model_name
        randens.train(x_train_projected, y_train, device)
        print("\nProjection + training time: --- %s seconds ---" % (time.time() - start_time))

        self.trained = True
        randens.save_classifier(relative_path=RESULTS)
        return randens

    def parallel_train(self, device):
        Parallel(n_jobs=self.n_proj)(
            delayed(_parallel_train)(dataset_name=self.dataset_name, test=self.test, n_proj=self.n_proj,
                                     proj_idx=proj_idx, size_proj=self.size_proj, proj_mode=self.projection_mode,
                                     device=device)
            for proj_idx in range(self.n_proj))
        self.trained = True


def _parallel_train(dataset_name, test, n_proj, proj_idx, size_proj, proj_mode, device):
    import tensorflow as tf
    g = tf.Graph()
    with g.as_default():
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)

        model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj,
                                       n_proj=n_proj, data_format=data_format, dataset_name=dataset_name,
                                       projection_mode=proj_mode, test=test)
        model.train_single_projection(x_train=x_train, y_train=y_train, device=device, proj_idx=proj_idx)
        del tf


def main(dataset_name, test, n_proj, size_proj, proj_mode):

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)
    model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj, n_proj=n_proj,
                           data_format=data_format, dataset_name=dataset_name, projection_mode=proj_mode, test=test)
    model.parallel_train(device=device)

    # ======== buggy =========
    del model
    model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj, n_proj=n_proj,
                                   data_format=data_format, dataset_name=dataset_name, projection_mode=proj_mode,
                                   test=test)
    model.load_classifier(relative_path=RESULTS)
    model.evaluate(x_test, y_test)
    for attack in ['fgsm','pgd','deepfool','carlini']:
        x_test_adv = model.load_adversaries(attack=attack, eps=0.5)
        model.evaluate(x_test_adv, y_test)


if __name__ == "__main__":

    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        n_proj = int(sys.argv[3])
        size_proj_list = list(map(int, sys.argv[4].strip('[]').split(',')))
        projection_mode = sys.argv[5]
        device = sys.argv[6]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        n_proj = input("\nChoose the number of projections (int): . ")
        size_proj_list = input("\nChoose size for the projection (list): ")
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")
        device = input("\nChoose a device (cpu/gpu): ")

    for size_proj in size_proj_list:
        main(dataset_name=dataset_name, test=test, n_proj=n_proj, size_proj=size_proj, proj_mode=projection_mode)

