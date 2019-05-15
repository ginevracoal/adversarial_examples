"""
This model computes random projections of the input points in a lower dimensional space and performs classification
separately on each projection, then it returns an ensemble classification on the original input data.
"""

import time
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod
from baseline_convnet import BaselineConvnet
from keras.models import load_model
from sklearn.random_projection import GaussianRandomProjection
from utils import *
import pickle as pkl


SAVE_MODEL = False
TEST = False
MODEL_NAME = "random_ensemble"
TRAINED_BASELINE = "IBM-art/mnist_cnn_original.h5"
TRAINED_MODELS = "../trained_models/"

BATCH_SIZE = 128
EPOCHS = 12
N_PROJECTIONS = 5
SIZE_PROJECTION = 8
SEED = 123

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
        self.projector = None
        self.trained = False

    def train(self, x_train, y_train, batch_size, epochs):
        """
        Trains the baseline model over `n_proj` random projections of the training data whose input shape is
        `(size_proj, size_proj, 1)`.

        :param x_train:
        :param y_train:
        :param batch_size:
        :param epochs:
        :return: list of n_proj trained models
        """

        print("\nGaussianRandomProjector seed = ", SEED)
        self.projector = GaussianRandomProjection(n_components=self.size_proj * self.size_proj, random_state=SEED)
        x_train_projected = compute_projections(x_train, self.projector, n_proj=self.n_proj, size_proj=self.size_proj)

        # use the same model for all trainings
        convNet = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)

        # train n_proj classifiers on different training data
        classifier = [convNet.train(x_train_projected[i], y_train, batch_size=batch_size, epochs=epochs) for i in
                      range(len(x_train_projected))]

        self.trained = True

        return classifier

    @staticmethod
    def _ensemble_classifier(classifiers, projected_test_data):
        """
        :param classifiers: list of `n_proj` different GaussianRandomProjection objects
        :param projected_test_data: array of test data projected on the different `n_proj` training random directions
        (`size_proj` directions for each projection)
        :return: sum of all the predicted probabilities among each class for the `n_proj` classifiers
        """
        predictions = np.array([classifier.predict(projected_test_data[i]) for i, classifier in enumerate(classifiers)])
        summed_predictions = np.sum(predictions, axis=0)
        return summed_predictions

    def predict(self, classifier, x_test):
        """
        Compute the average prediction over the trained models.

        :param classifier: list of trained classifiers over different projections
        :param x_test: list of projected test data
        :return:
        """
        if self.projector is None:
            raise ValueError("There is no projector available. Train the model first.")

        x_test_projected = compute_projections(x_test, projector=self.projector,
                                               n_proj=self.n_proj, size_proj=self.size_proj)

        predictions = self._ensemble_classifier(classifier, x_test_projected)

        return predictions

    def old_evaluate_adversaries(self, classifier, x_test, y_test, method='fgsm', adversaries_path=None):
        """
        Evaluates the trained model against FGSM and prints the number of misclassifications. Here FGSM is applied to
        the baseline classifier.

        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        :return:
        x_test_pred: test set predictions
        x_test_adv: adversarial perturbations of test data
        x_test_adv_pred: adversarial test set predictions
        """
        convNet = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)
        baseline_classifier = convNet.load_classifier(TRAINED_BASELINE)

        if method == 'fgsm':
            print("\nAdversarial evaluation using FGSM method.")
            attacker = FastGradientMethod(baseline_classifier, eps=0.5)
            x_test_adv = attacker.generate(x_test)

        elif method == 'deepfool':
            print("\nAdversarial evaluation using DeepFool method.")
            with open('../data/mnist_x_test_deepfool.pkl', 'rb') as f:
                u = pkl._Unpickler(f)
                u.encoding = 'latin1'
                x_test_adv = u.load()

        # evaluate the performance on the list of trained classifiers
        x_test_adv_pred = np.argmax(self.predict(classifier, x_test_adv), axis=1)
        nb_correct_adv_pred = np.sum(x_test_adv_pred == np.argmax(y_test, axis=1))

        print("\nAdversarial test data:")
        print("Correctly classified: {}".format(nb_correct_adv_pred))
        print("Incorrectly classified: {}".format(len(x_test) - nb_correct_adv_pred))

        acc = np.sum(x_test_adv_pred == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print("Adversarial accuracy: %.2f%%" % (acc * 100))

        return x_test_adv, x_test_adv_pred

    def _generate_adversaries(self, classifier, x, method='fgsm', adversaries_path=None):
        """ Adversaries are generated on the baseline classifier """
        convNet = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)
        baseline_classifier = convNet.load_classifier(TRAINED_BASELINE)
        x_adv = convNet._generate_adversaries(baseline_classifier, x, method=method, adversaries_path=adversaries_path)
        return x_adv

    def save_model(self, classifier, model_name):
        if self.trained:
            for i, proj_classifier in enumerate(classifier):
                proj_classifier.save(filename=model_name + "_" + str(i) + ".h5",
                                     path=TRAINED_MODELS + time.strftime('%Y-%m-%d'))

    def load_classifier(self, relative_path, model_name=MODEL_NAME):
        """
        Loads a pretrained classifier and sets the projector with the training seed.
        :param relative_path: here refers to the relative path of the folder containing the list of trained classifiers
        :param model_name: name of the model used when files were saved
        :return: list of trained classifiers
        """
        # load all trained models
        trained_models = [load_model(relative_path + model_name + "_" + str(i) + ".h5") for i in range(self.n_proj)]
        classifiers = [KerasClassifier((MIN, MAX), model, use_logits=False) for model in trained_models]

        # build a random projector with the same original seed
        self.projector = GaussianRandomProjection(n_components=self.size_proj * self.size_proj, random_state=SEED)

        return classifiers, self.projector


def main():

    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_mnist()

    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes,
                           n_proj=N_PROJECTIONS, size_proj=SIZE_PROJECTION)

    #classifier = model.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    classifier, projector = model.load_classifier(relative_path="../trained_models/random_ensemble/2019-05-14/")

    model.evaluate_test(classifier, x_test, y_test)
    x_test_adv, x_test_adv_pred = model.evaluate_adversaries(classifier, x_test, y_test, method='deepfool')

    # bug
    print(np.argwhere(np.isnan(x_test_adv)))
    exit()

    if SAVE_MODEL is True:
        model.save_model(classifier=classifier, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
