# -*- coding: utf-8 -*-

"""
This model computes random projections of the input points in a lower dimensional space and performs classification
separately on each projection, then it returns an ensemble classification on the original input data.
"""

from art.classifiers import KerasClassifier
from baseline_convnet import BaselineConvnet
from keras.models import load_model
from utils import *

# settings
TEST = True
SIZE_PROJECTION = 8
N_PROJECTIONS = 10

# defaults
MODEL_NAME = "random_ensemble"
TRAINED_MODELS = "../trained_models/"
TRAINED_BASELINE = TRAINED_MODELS+"IBM-art/mnist_cnn_original.h5"
DEEPFOOL_PATH = "../data/mnist_x_test_deepfool.pkl"
DATA_PATH = "../data/"
RESULTS = "../results/"
BATCH_SIZE = 128
EPOCHS = 12
MIN = 0
MAX = 255


class RandomEnsemble(BaselineConvnet):
    """
    Classifies `n_proj` random projections of the training data in a lower dimensional space (whose dimension is
    `size_proj`^2), then classifies the original high dimensional data with a voting technique.
    """
    def __init__(self, input_shape, num_classes, n_proj, size_proj):
        """
        Extends BaselineConvnet initializer with additional informations about the projections.
        :param input_shape: full dimension input data shape
        :param num_classes: number of classes
        :param n_proj: number of random projections
        :param size_proj: size of a random projection
        """

        if size_proj > input_shape[1]:
            raise ValueError("The number of projections has to be lower than the image size.")

        super(RandomEnsemble, self).__init__(input_shape, num_classes)
        self.input_shape = (size_proj, size_proj, 1)
        self.n_proj = n_proj
        self.size_proj = size_proj
        self.random_seeds = np.array([123, 45, 180, 172, 61, 63, 70, 83, 115, 67])
        self.projector = None
        self.projectors_params = None
        self.trained = False

    def train(self, x_train, y_train, batch_size, epochs):
        """
        Trains the baseline model over `n_proj` random projections of the training data whose input shape is
        `(size_proj, size_proj, 1)`.

        :param x_train:
        :param y_train:
        :param batch_size:
        :param epochs:
        :return: list of n_proj trained models, which are art.KerasClassifier fitted objects
        """

        x_train_projected = compute_projections(x_train, random_seeds=self.random_seeds,
                                                n_proj=self.n_proj, size_proj=self.size_proj)
        # use the same model for all trainings
        baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)
        # train n_proj classifiers on different training data
        classifiers = [baseline.train(x_train_projected[i], y_train, batch_size=batch_size, epochs=epochs)
                       for i in range(self.n_proj)]
        self.trained = True
        return classifiers

    @staticmethod
    def _ensemble_classifier(classifiers, projected_data):
        """
        :param classifiers: list of `n_proj` different GaussianRandomProjection objects
        :param projected_test_data: array of test data projected on the different `n_proj` training random directions
        (`size_proj` directions for each projection)
        :return: sum of all the predicted probabilities among each class for the `n_proj` classifiers
        """
        predictions = np.array([classifier.predict(projected_data[i]) for i, classifier in enumerate(classifiers)])
        # sum the probabilities across all predictors
        summed_predictions = np.sum(predictions, axis=0)
        #print("\nPredictions on the first element:\n", predictions[:, 0])
        return summed_predictions

    def predict(self, classifier, x_test):
        """
        Compute the average prediction over the trained models.

        :param classifier: list of trained classifiers over different projections
        :param x_test: list of projected test data
        :return:
        """
        x_test_projected = compute_projections(x_test, random_seeds=self.random_seeds,
                                               n_proj=self.n_proj, size_proj=self.size_proj)
        predictions = self._ensemble_classifier(classifier, x_test_projected)
        return predictions

    def _generate_adversaries(self, classifier, x, y, method='fgsm', adversaries_path=None):
        """ Adversaries are generated on the baseline classifier """
        baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)
        baseline_classifier = baseline.load_classifier(TRAINED_BASELINE)
        x_adv = baseline._generate_adversaries(baseline_classifier, x, y, method=method,
                                               adversaries_path=adversaries_path)
        return x_adv

    def save_model(self, classifier, model_name):
        if self.trained:
            for i, proj_classifier in enumerate(classifier):
                proj_classifier.save(filename=model_name + "_" + str(self.random_seeds[i]) + ".h5",
                                     path=RESULTS + time.strftime('%Y-%m-%d'))
        else:
            raise ValueError("Model has not been fitted!")

    def load_classifier(self, relative_path, model_name):
        """
        Loads a pretrained classifier and sets the projector with the training seed.
        :param relative_path: here refers to the relative path of the folder containing the list of trained classifiers
        :param model_name: name of the model used when files were saved
        :return: list of trained classifiers
        """
        # load all trained models
        trained_models = [load_model(relative_path + model_name + "_" + str(seed) + ".h5")
                          for seed in self.random_seeds[:self.n_proj]]
        classifiers = [KerasClassifier((MIN, MAX), model, use_logits=False) for model in trained_models]
        return classifiers


def main():

    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_mnist(test=TEST)

    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes,
                           n_proj=N_PROJECTIONS, size_proj=SIZE_PROJECTION)

    classifier = model.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    model.evaluate_test(classifier, x_test, y_test)
    #model.save_model(classifier=classifier, model_name=MODEL_NAME)

    classifier = model.load_classifier(
        relative_path=TRAINED_MODELS+"random_ensemble/random_ensemble_proj10_size8_new/",
        model_name=MODEL_NAME)

    model.evaluate_test(classifier, x_test, y_test)

    model.evaluate_adversaries(classifier, x_test, y_test, method='fgsm')
    model.evaluate_adversaries(classifier, x_test, y_test, method='deepfool', adversaries_path='../data/mnist_x_test_deepfool.pkl')
    model.evaluate_adversaries(classifier, x_test, y_test, method='projected_gradient', adversaries_path='../data/mnist_x_test_projected_gradient.pkl')

    # model.evaluate_adversaries(classifier, x_test, y_test, method='virtual_adversarial')
    # model.evaluate_adversaries(classifier, x_test, y_test, method='carlini_l2')

    # save_to_pickle(data=x_test_deepfool, filename="mnist_x_test_deepfool.pkl")


if __name__ == "__main__":
    main()
