# -*- coding: utf-8 -*-

from random_regularizer import *

class Ensemble_Regularizer(RandomRegularizer):
    def __init__(self, ensemble_size, input_shape, num_classes, data_format, dataset_name, lam, projection_mode, test):
        self.ensemble_size = ensemble_size
        super(Ensemble_Regularizer, self).__init__(input_shape, num_classes, data_format, dataset_name, lam,
                                                   projection_mode, test)

    def save_classifier(self, classifiers):
        """
        Saves all the ensemble classifiers separately using their index in filenames.
        :param classifiers: list of trained classifiers.
        """
        raise NotImplementedError

    def load_classifier(self, relative_path):
        """
        Loads trained ensemble classifiers in a list.
        :param relative_path: relative path of folder containing trained models.
        :return: list of loaded classifiers.
        """
        raise NotImplementedError


def main(dataset_name, test, ensemble_size, projection_mode):
    """
    :param dataset_name: choose between "mnist" and "cifar"
    :param test: if True only takes the first 100 samples
    :param ensemble_size: number of models for the ensemble
    :param projection_mode: method for computing projections on RGB images
    """

    # === initialize === #
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)

    model = Ensemble_Regularizer(ensemble_size=ensemble_size, input_shape=input_shape, num_classes=num_classes,
                                 projection_mode=projection_mode, data_format=data_format, dataset_name=dataset_name,
                                 lam=lam, test=test)
    # === train === #
    classifier = model.train(x_train, y_train)
    # model.save_classifier(classifier)
    # classifier = model.load_classifier(relative_path=TRAINED_MODELS)

    # === evaluate === #
    model.evaluate(classifier=classifier, x=x_test, y=y_test)

    x_test_adv = model.load_adversaries(dataset_name=dataset_name,attack=attack, eps=eps, test=test)
    model.evaluate(classifier=classifier, x=x_test_adv, y=y_test)


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        ensemble_size = int(sys.argv[3])
        projection_mode = sys.argv[4]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        ensemble_size = input("\nChoose the ensemble size (integer value): ")
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")

    main(dataset_name, test, ensemble_size, projection_mode)



