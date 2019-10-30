# -*- coding: utf-8 -*-

from random_regularizer import *
from tensorflow.python.client import device_lib
# import multiprocessing
# from joblib import Parallel, delayed


class EnsembleRegularizer(RandomRegularizer):
    """
    Performs an ensemble classification using different random_regularizer models.
    """

    def __init__(self, ensemble_size, input_shape, num_classes, data_format, dataset_name, lam, projection_mode, test,
                 init_seed=1):
        self.ensemble_size = ensemble_size
        self.ensemble_classifiers = None
        super(EnsembleRegularizer, self).__init__(input_shape, num_classes, data_format, dataset_name, lam,
                                                  projection_mode, test, init_seed)

    def train(self, x_train, y_train, device="cpu"):
        """
        Trains different random_regularizer models over random projections of the training data.
        :param x_train: training data
        :param y_train: training labels
        :param device: code execution device (cpu/gpu)
        :return: list of `ensemble_size` trained models
        """

        start_time = time.time()

        # if parallel:
        #     num_cores = multiprocessing.cpu_count()
        #     parallel_train = lambda model: model.train(x_train, y_train, device)
        #     models = []
        #     for i in range(self.ensemble_size):
        #         randreg = RandomRegularizer(input_shape=self.input_shape, num_classes=self.num_classes,
        #                                     data_format=self.data_format, dataset_name=self.dataset_name, lam=self.lam,
        #                                     projection_mode=self.projection_mode, test=self.test, init_seed=i)
        #         models.append(randreg)
        #     classifiers = Parallel(n_jobs=num_cores)(delayed(parallel_train)(m) for m in models)
        classifiers = []
        for i in range(self.ensemble_size):
            randreg = RandomRegularizer(input_shape=self.input_shape, num_classes=self.num_classes,
                                        data_format=self.data_format, dataset_name=self.dataset_name, lam=self.lam,
                                        projection_mode=self.projection_mode, test=self.test, init_seed=i)
            classifiers.append(randreg.train(x_train, y_train, device))
            del randreg

        print("\nTraining time for Ensemble Regularizer with ensemble_size = ", str(self.ensemble_size),
              " : --- %s seconds ---" % (time.time() - start_time))

        self.ensemble_classifiers = classifiers

    def predict(self, x, **kwargs):
        """
        Computes the ensemble prediction.
        :param x: input data
        :return: ensemble predictions on the input data
        """
        # compute predictions from different models of the ensemble
        predictions_list = np.array([classifier.predict(x) for classifier in self.ensemble_classifiers])
        # sum the probabilities across all predictors
        ensemble_predictions = np.sum(predictions_list, axis=0)
        return ensemble_predictions

    def save_classifier(self, relative_path=RESULTS + time.strftime('%Y-%m-%d') + "/"):
        """
        Saves all the ensemble classifiers separately using their index in filenames.
        :relative_path: path of folder containing the trained model
        """
        if self.ensemble_classifiers:
            for classifier in self.ensemble_classifiers:
                classifier.save_classifier()
        else:
            raise AttributeError("\nModel has not been trained yet.")

    def load_classifier(self, relative_path):
        """
        Loads trained ensemble classifiers in a list.
        :param relative_path: relative path of folder containing trained models.
        :return: list of loaded classifiers.
        """
        start_time = time.time()
        classifiers = []
        for i in range(self.ensemble_size):
            randreg = RandomRegularizer(input_shape=self.input_shape, num_classes=self.num_classes,
                                        data_format=self.data_format, dataset_name=self.dataset_name, lam=self.lam,
                                        projection_mode=self.projection_mode, test=self.test, init_seed=i)
            classifiers.append(randreg.load_classifier(relative_path))
            del randreg
        print("\nLoading time: --- %s seconds ---" % (time.time() - start_time))
        self.ensemble_classifiers = classifiers


def main(dataset_name, test, ensemble_size, projection_mode, lam, device):
    """
    :param dataset_name: choose between "mnist" and "cifar"
    :param test: if True only takes the first 100 samples
    :param ensemble_size: number of models for the ensemble
    :param projection_mode: method for computing projections on RGB images
    :param lam: lambda regularization weight parameter
    :param device: code execution device (cpu/gpu)
    """

    # === initialize === #
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)

    model = EnsembleRegularizer(ensemble_size=ensemble_size, input_shape=input_shape, num_classes=num_classes,
                                projection_mode=projection_mode, data_format=data_format, dataset_name=dataset_name,
                                lam=lam, test=test)
    # === train === #
    # model.train(x_train, y_train, device)
    # model.save_classifier()
    # model.load_classifier(relative_path=TRAINED_MODELS)
    model.load_classifier(relative_path=RESULTS + time.strftime('%Y-%m-%d') + "/")

    # === evaluate === #
    model.evaluate(x=x_test, y=y_test)

    # set max norm for adversarial perturbations
    eps = 0.3
    for attack in ['fgsm','pgd','deepfool','carlini']:
        x_test_adv = model.load_adversaries(dataset_name=dataset_name,attack=attack, eps=eps, test=test)
        model.evaluate(x=x_test_adv, y=y_test)


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        ensemble_size = int(sys.argv[3])
        projection_mode = sys.argv[4]
        lam = float(sys.argv[5])
        device = sys.argv[6]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        ensemble_size = input("\nChoose the ensemble size (integer value): ")
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")
        lam = float(input("\nChoose lambda regularization weight (type=float): "))
        device = input("\nChoose a device (cpu/gpu): ")

    main(dataset_name, test, ensemble_size, projection_mode, lam, device)



