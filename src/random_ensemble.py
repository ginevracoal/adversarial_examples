# -*- coding: utf-8 -*-

"""
This model computes random projections of the input points in a lower dimensional space and performs classification
separately on each projection, then it returns an ensemble classification on the original input data.
"""

from art.classifiers import KerasClassifier
from baseline_convnet import BaselineConvnet
from keras.models import load_model
from utils import *
import time
from keras import backend as K
import multiprocessing as mp

###############
# main() args #
###############
DATASET = "mnist"
TEST = True
ATTACK = ["fgsm"] #,"pgd","deepfool","carlini_linf"]
N_PROJECTIONS = [6] # default for training is [15], default for testing is [6,9,12,15]
SIZE_PROJECTIONS = [8]#,12,16,20]

####################
# default settings #
####################
MODEL_NAME = "random_ensemble"
ENSEMBLE_METHOD = "sum"  # possible methods: mode, sum
TRAINED_MODELS = "../trained_models/random_ensemble/"
DATA_PATH = "../data/"
RESULTS = "../results/"
BATCH_SIZE = 128
EPOCHS = 12
MIN = 0
MAX = 255
SEED = 123


class RandomEnsemble(BaselineConvnet):
    """
    Classifies `n_proj` random projections of the training data in a lower dimensional space (whose dimension is
    `size_proj`^2), then classifies the original high dimensional data with an ensemble classifier, summing up the
    probabilities from the single projections.
    """
    def __init__(self, input_shape, num_classes, n_proj, size_proj, data_format, dataset_name):
        """
        Extends BaselineConvnet initializer with additional informations about the projections.
        :param n_proj: number of random projections
        :param size_proj: size of a random projection
        """

        if size_proj > input_shape[1]:
            raise ValueError("The number of projections has to be lower than the image size.")

        super(RandomEnsemble, self).__init__(input_shape, num_classes, data_format, dataset_name)
        self.input_shape = (size_proj, size_proj, 1) #input_shape[2])
        self.n_proj = n_proj
        self.size_proj = size_proj
        # the model is currently implemented on 15 projections max
        self.random_seed = np.array([123, 45, 180, 172, 61, 63, 70, 83, 115, 67, 56, 133, 12, 198, 156])  # np.repeat(123, 10)
        #self.projector = None
        #self.projectors_params = None
        self.trained = False
        self.training_time = 0
        #self.data_format = data_format
        # todo: set ensemble method here

        print("\n === RandEns model ( n_proj =", self.n_proj, ", size_proj =", self.size_proj, ") ===")

    ###################
    # serial training #
    ###################
    def serial_train(self, x_train, y_train, batch_size, epochs):
        """
        Trains the baseline model over `n_proj` random projections of the training data whose input shape is
        `(size_proj, size_proj, 1)`. This function is currently unused.

        :param x_train:
        :param y_train:
        :param batch_size:
        :param epochs:
        :return: list of n_proj trained models, which are art.KerasClassifier fitted objects
        """

        start_time = time.time()
        x_train_projected = compute_projections(x_train, random_seed=self.random_seed,
                                                n_proj=self.n_proj, size_proj=self.size_proj)

        classifiers = []
        for i in range(self.n_proj):
            # use the same model architecture (not weights) for all trainings
            baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                       data_format=self.data_format, dataset_name=self.dataset_name)
            # train n_proj classifiers on different training data
            classifiers.append(baseline.train(x_train_projected[i], y_train, batch_size=batch_size, epochs=epochs))
            del baseline

        print("\nTraining time for model (n_proj=", str(self.n_proj), ", size_proj=", str(self.size_proj),
              "): --- %s seconds ---" % (time.time() - start_time))

        self.trained = True
        return classifiers

    #####################
    # parallel training #
    #####################
    def train_save_projection(self, x_train_projected, y_train, batch_size, epochs, idx):
        """ Trains a single projection of the ensemble classifier and saves the model in current day results folder."""
        K.clear_session()
        # use the same model architecture (not weights) for all trainings
        baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                   dataset_name=self.dataset_name, data_format=self.data_format)
        # train n_proj classifiers on different training data
        classifier = baseline.train(x_train_projected, y_train, batch_size=batch_size, epochs=epochs)

        start = time.time()
        classifier.save(filename=MODEL_NAME + "_size="+ str(self.size_proj) + "_" + str(self.random_seed[idx]) + ".h5",
                             path=RESULTS + time.strftime('%Y-%m-%d'))
        saving_time = time.time() - start

        self.training_time -= saving_time

        return classifier

    def train(self, x_train, y_train, batch_size, epochs):
        """
        Trains the baseline model over `n_proj` random projections of the training data whose input shape is
        `(size_proj, size_proj, 1)` and parallelizes training over the different projections.

        :param x_train:
        :param y_train:
        :param batch_size:
        :param epochs:
        :return: list of n_proj trained models, which are art.KerasClassifier fitted objects
        """

        start = time.time()
        x_train_projected = compute_projections(x_train, random_seed=self.random_seed,
                                                n_proj=self.n_proj, size_proj=self.size_proj)

        # Define an output queue
        #output = mp.Queue()
        # Setup a list of processes that we want to run
        processes = [mp.Process(target=self.train_save_projection,
                                args=(x_train_projected[i], y_train, batch_size, epochs, i)) for i in range(self.n_proj)]
        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        self.training_time += + time.time() - start

        # Get process results from the output queue
        #classifiers = [output.get() for p in processes]

        print("\nParallel training time for model ( n_proj =", str(self.n_proj), ", size_proj =", str(self.size_proj),
              "): --- %s seconds ---" % (self.training_time))
        classifiers = self.load_classifier(relative_path=RESULTS+time.strftime('%Y-%m-%d')+"/",
                                           model_name="random_ensemble")

        self.trained = True

        return classifiers
    #################

    @staticmethod
    def _sum_ensemble_classifier(classifiers, projected_data):
        """
        :param classifiers: list of `n_proj` different GaussianRandomProjection objects
        :param projected_data: array of test data projected on the different `n_proj` training random directions
        (`size_proj` directions for each projection)
        :return: sum of all the predicted probabilities among each class for the `n_proj` classifiers

        ```
        Predictions on the first element by each single classifier
        predictions[:, 0] =
         [[0.04745529 0.00188083 0.01035858 0.21188359 0.00125252 0.44483757
          0.00074033 0.13916749 0.01394993 0.1284739 ]
         [0.00259137 0.00002327 0.48114488 0.42658636 0.00003032 0.01012747
          0.00002206 0.03735029 0.02623402 0.0158899 ]
         [0.00000004 0.00000041 0.00000277 0.00001737 0.00000067 0.00000228
          0.         0.9995009  0.00000014 0.0004754 ]]

        Ensemble prediction vector on the first element
        summed_predictions[0] =
         [0.05004669 0.00190451 0.49150622 0.6384873  0.0012835  0.45496735
         0.0007624  1.1760187  0.04018409 0.14483918]
        """
        # compute predictions for each projection
        proj_predictions = np.array([classifier.predict(projected_data[i]) for i, classifier in enumerate(classifiers)])
        # sum the probabilities across all predictors
        predictions = np.sum(proj_predictions, axis=0)
        return predictions

    @staticmethod
    def _mode_ensemble_classifier(classifiers, projected_data):
        """
        :param classifiers: list of `n_proj` different GaussianRandomProjection objects
        :param projected_data: array of test data projected on the different `n_proj` training random directions
        (`size_proj` directions for each projection)
        :return: compute the argmax of the probability vectors and then, for each points, choose the mode over all
        classes as the predicted label

        ```
        Computing random projections.
        Input shape:  (100, 28, 28, 1)
        Projected data shape: (3, 100, 8, 8, 1)

        Predictions on the first element by each single classifier
        predictions[:, 0] =
        [[0.09603461 0.08185963 0.03264992 0.07047556 0.2478332  0.03418195
          0.13880958 0.19712913 0.04649669 0.05452974]
         [0.0687536  0.14464664 0.0766349  0.09082405 0.1066305  0.01555605
          0.03265413 0.12625733 0.14203466 0.19600812]
         [0.16379683 0.0895557  0.07057846 0.09945401 0.25141633 0.04555665
          0.08481159 0.0610559  0.06158736 0.07218721]]

        argmax_predictions[:, 0] =
        [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
         [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]

        Ensemble prediction vector on the first element
        summed_predictions[0] =
        [0.         0.         0.         0.         0.66666667 0.
         0.         0.         0.         0.33333333]
        ```
        """

        # compute predictions for each projection
        proj_predictions = np.array([classifier.predict(projected_data[i]) for i, classifier in enumerate(classifiers)])

        # convert probability vectors into target vectors
        argmax_predictions = np.zeros(proj_predictions.shape)
        for i in range(proj_predictions.shape[0]):
            idx = np.argmax(proj_predictions[i], axis=-1)
            argmax_predictions[i, np.arange(proj_predictions.shape[1]), idx] = 1
        # sum the probabilities across all predictors
        predictions = np.sum(argmax_predictions, axis=0)
        # normalize
        predictions = predictions / predictions.sum(axis=1)[:, None]

        return predictions

    def predict(self, classifiers, data, method=ENSEMBLE_METHOD, *args, **kwargs):
        """
        Compute the average prediction over the trained models.

        :param classifiers: list of trained classifiers over different projections
        :param data: input data
        :param method: ensemble method chosen. Only sum and mode are currently implemented.
        :return: final predictions for the input data
        """
        projected_data = compute_projections(data, random_seed=self.random_seed,
                                             n_proj=self.n_proj, size_proj=self.size_proj)
        predictions = None
        if method == 'sum':
            predictions = self._sum_ensemble_classifier(classifiers, projected_data)
        elif method == 'mode':
            predictions = self._mode_ensemble_classifier(classifiers, projected_data)

        return predictions

    def evaluate_test_projections(self, classifiers, x_test, y_test):
        """
        Performs a test evaluation on each projected version of the data and also on the final predictions.
        :param classifiers: list of trained classifiers over different projections
        :param x_test: test data
        :param y_test: test labels
        :return: y_test predictions
        """
        print("\nTesting infos:\nx_test.shape = ", x_test.shape, "\ny_test.shape = ", y_test.shape, "\n")

        x_test_proj = compute_projections(x_test, random_seed=self.random_seed,
                                          n_proj=self.n_proj, size_proj=self.size_proj)
        y_test_pred = np.argmax(self.predict(classifiers, x_test, method=ENSEMBLE_METHOD), axis=1)

        # evaluate each classifier on its projected test set
        baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)
        for i, classifier in enumerate(classifiers):
            print("\nTest evaluation on projection ", self.random_seed[i])  # i
            baseline.evaluate_test(classifier, x_test_proj[i], y_test)

        # final classifier evaluation on the original test set
        print("\nFinal test evaluation")
        super(BaselineConvnet, self).evaluate_test(classifiers, x_test, y_test)

        return y_test_pred

    def _generate_adversaries(self, classifier, x, y, method, dataset_name, adversaries_path=None, test=False, *args, **kwargs):
        """ Adversaries are generated on the baseline classifier """
        baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                   data_format=self.data_format, dataset_name=self.dataset_name)
        baseline_classifier = baseline.load_classifier("../trained_models/baseline/"+dataset_name+"_baseline.h5")
        x_adv = baseline._generate_adversaries(baseline_classifier, x, y, method=method, dataset_name=dataset_name,
                                               adversaries_path=adversaries_path, test=test)
        return x_adv

    def save_model(self, classifier, model_name):
        if self.trained:
            for i, proj_classifier in enumerate(classifier):
                proj_classifier.save(filename=model_name + "_" + str(self.random_seed[i]) + ".h5",
                                     path=RESULTS + time.strftime('%Y-%m-%d'))
        else:
            raise ValueError("Model has not been fitted!")

    def load_classifier(self, relative_path, model_name="random_ensemble"):
        """
        Loads a pretrained classifier and sets the projector with the training seed.
        :param relative_path: here refers to the relative path of the folder containing the list of trained classifiers
        :param model_name: name of the model used when files were saved
        :return: list of trained classifiers
        """
        start_time = time.time()
        # load all trained models
        trained_models = [load_model(relative_path+model_name+"_size=" + str(self.size_proj)+"_"+str(seed)+".h5")
                          for seed in self.random_seed[:self.n_proj]]
        print("\nLoading time: --- %s seconds ---" % (time.time() - start_time))

        classifiers = [KerasClassifier((MIN, MAX), model, use_logits=False) for model in trained_models]
        return classifiers


########
# MAIN #
########

def main(dataset, test, attack, n_proj, size_proj):
    """
    :param dataset: choose between "mnist" and "cifar"
    :param test: if True only takes 100 samples
    :param attack: choose between "fgsm", "pgd", "deepfool","carlini_linf", "virtual", "newtonfool"
    :param n_proj: number of projections. Trained models used values: 6, 9, 12, 15.
    :param size_proj: size of each projection. You can currently load models with values: 8, 12, 16, 20.
    """

    # === load data === #
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset, test=test)

    # === train === #
    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes,
                           n_proj=n_proj, size_proj=size_proj, data_format=data_format, dataset_name=dataset)

    classifier = model.train(x_train, y_train, batch_size=model.batch_size, epochs=model.epochs)
    # model.save_model(classifier=classifier, model_name=dataset+"_random_ensemble_size=" + str(model.size_proj))

    # === load classifier === #
    #relpath = dataset + "_random_ensemble_sum_size=" + str(model.size_proj) + "/"
    #classifier = model.load_classifier(relative_path=TRAINED_MODELS + relpath, model_name=MODEL_NAME)

    # === adversarial train === #
    #robust_classifier = model.adversarial_train(classifier, x_train, y_train, x_test, y_test, dataset_name=dataset,
    #                                            batch_size=model.batch_size, epochs=model.epochs, method=attack, test=test)
    #model.save_model(classifier=robust_classifier, model_name=dataset + "_" + str(attack) +
    #                 "_robust_random_ensemble_size=" + str(model.size_proj))

    # === evaluate === #
    model.evaluate_test(classifier, x_test, y_test)
    #model.evaluate_test(robust_classifier, x_test, y_test)

    #model.evaluate_adversaries(classifier, x_test, y_test, method=attack, test=test, dataset_name=dataset)
    #                           adversaries_path='../data/'+dataset+'_x_test_' + attack + '.pkl')
    #model.evaluate_adversaries(robust_classifier, x_test, y_test, method=attack, dataset_name=dataset, test=test)


if __name__ == "__main__":

    for n_proj in N_PROJECTIONS:
        for size_proj in SIZE_PROJECTIONS:
            K.clear_session()
            main(DATASET, TEST, ATTACK, n_proj, size_proj)
