import numpy as np
import time
from keras.models import load_model
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod
from utils import *


TRAINED_MODELS = "../trained_models/"


class AdversarialClassifier:
    """
    Keras Classifier base class
    """

    def __init__(self, input_shape, num_classes, regularizer=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._set_layers()
        self.regularizer = regularizer
        self.classifier = None

    def _set_layers(self):
        """
        defines the layers structure for the classifier
        :return: model
        """
        raise NotImplementedError

    def train(self, x_train, y_train, batch_size, epochs):
        """
        Trains the model using art.KerasClassifier wrapper, which then allows to easily train adversaries
        using the same package.
        :param x_train:
        :param y_train:
        :param batch_size:
        :param epochs:
        :return: trained classifier
        """
        classifier = KerasClassifier((MIN, MAX), model=self.model, use_logits=False)
        classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs)
        self.classifier = classifier
        return classifier

    def predict(self, classifier, x_test):
        #return self.classifier.predict(x_test)
        return classifier.predict(x_test)

    def evaluate_test(self, classifier, x_test, y_test):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        """
        preds = np.argmax(self.predict(classifier, x_test), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print("\nTest accuracy: %.2f%%" % (acc * 100))

    def evaluate_adversaries(self, classifier, x_test, y_test):
        """
        Evaluates the trained model against FGSM and prints the number of misclassifications.
        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        :return:
        x_test_pred: test set predictions
        x_test_adv: adversarial perturbations of test data
        x_test_adv_pred: adversarial test set predictions
        """
        # TODO: implementare tutto con questa sintassi, che è quella corretta
        #x_test_pred = np.argmax(self.classifier.predict(x_test), axis=1)
        x_test_pred = np.argmax(self.predict(classifier, x_test), axis=1)
        correct_preds = np.sum(x_test_pred == np.argmax(y_test, axis=1))

        print("\nOriginal test data:")
        print("Correctly classified: {}".format(correct_preds))
        print("Incorrectly classified: {}".format(len(x_test) - correct_preds))

        # generate adversarial examples using FGSM
        attacker = FastGradientMethod(classifier, eps=0.5)
        x_test_adv = attacker.generate(x_test)

        # evaluate the performance
        x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
        nb_correct_adv_pred = np.sum(x_test_adv_pred == np.argmax(y_test, axis=1))

        print("\nAdversarial test data:")
        print("Correctly classified: {}".format(nb_correct_adv_pred))
        print("Incorrectly classified: {}".format(len(x_test) - nb_correct_adv_pred))

        return x_test_pred, x_test_adv, x_test_adv_pred

    def save_model(self, classifier, model_name):
        """ Saves the trained model and adds the current datetime to the filename. """
        classifier.save(filename=model_name+"_"+time.strftime('%Y%m%d%H%M%S'), path=TRAINED_MODELS)

    def load_classifier(self, relative_path):
        """ Loads a pretrained classifier. """
        # load a trained model
        trained_model = load_model(TRAINED_MODELS+relative_path)
        classifier = KerasClassifier((MIN, MAX), trained_model, use_logits=False)
        #self.classifier = classifier
        return classifier
