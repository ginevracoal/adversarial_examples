# -*- coding: utf-8 -*-

import time
from keras.models import load_model
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod, DeepFool
# from art.metrics import clever_t, loss_sensitivity
from utils import *
import pickle as pkl


TRAINED_MODELS = "../trained_models/"
MIN = 0
MAX = 255


class AdversarialClassifier(object):
    """
    Keras Classifier base class
    """

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._set_layers()
        self.trained = False
        #self.classifier = None

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
        print("\nTraining infos:\nbatch_size = ", batch_size, "\nepochs =", epochs,
              "\nx_train.shape = ", x_train.shape, "\ny_train.shape = ", y_train.shape, "\n")

        classifier = KerasClassifier((MIN, MAX), model=self.model, use_logits=False)
        classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs)

        self.trained = True
        return classifier

    def predict(self, classifier, x):
        """
        This method is needed for calling the method predict on other objects than keras classifiers in the derived
        classes.
        :param classifier: trained classifier
        :param x: input data
        :return: predictions
        """
        return classifier.predict(x)

    def evaluate_test(self, classifier, x_test, y_test):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        :return: x_test predictions
        """
        print("\nTesting infos:\nx_test.shape = ", x_test.shape, "\ny_test.shape = ", y_test.shape, "\n")

        x_test_pred = np.argmax(self.predict(classifier, x_test), axis=1)
        correct_preds = np.sum(x_test_pred == np.argmax(y_test, axis=1))

        print("\nOriginal test data.")
        print("Correctly classified: {}".format(correct_preds))
        print("Incorrectly classified: {}".format(len(x_test) - correct_preds))

        acc = np.sum(x_test_pred == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print("Test accuracy: %.2f%%" % (acc * 100))

        return x_test_pred

    def _generate_adversaries(self, classifier, x, method='fgsm', adversaries_path=None):
        """
        Generates adversaries on the input data x using a given method or loads saved data if available.

        :param classifier: trained classifier
        :param x: input data
        :param method: art.attack method
        :param adversaries_path: path of saved pickle data
        :return: adversarially perturbed data
        """
        if method == 'fgsm':
            print("\nAdversarial evaluation using FGSM method.")
            attacker = FastGradientMethod(classifier, eps=0.5)
            x_adv = attacker.generate(x)
        elif method == 'deepfool':
            if adversaries_path is None:
                print("\nAdversarial evaluation using DeepFool method.")
                attacker = DeepFool(classifier)
                x_adv = attacker.generate(x)
            else:
                with open(adversaries_path, 'rb') as f:
                    u = pkl._Unpickler(f)
                    u.encoding = 'latin1'
                    x_adv = u.load()
        return x_adv

    def evaluate_adversaries(self, classifier, x_test, y_test, method='fgsm', adversaries_path=None):
        """
        Evaluates the trained model against FGSM and prints the number of misclassifications.
        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        :param method: art.attack method
        :param adversaries_path: path of saved pickle data
        :return:
        x_test_pred: test set predictions
        x_test_adv: adversarial perturbations of test data
        x_test_adv_pred: adversarial test set predictions
        """

        # generate adversaries on the test set
        x_test_adv = self._generate_adversaries(classifier, x_test, method=method, adversaries_path=adversaries_path)

        # evaluate the performance on the adversarial test set
        x_test_adv_pred = np.argmax(self.predict(classifier, x_test_adv), axis=1)
        nb_correct_adv_pred = np.sum(x_test_adv_pred == np.argmax(y_test, axis=1))

        print("\nAdversarial test data.")
        print("Correctly classified: {}".format(nb_correct_adv_pred))
        print("Incorrectly classified: {}".format(len(x_test) - nb_correct_adv_pred))

        acc = np.sum(x_test_adv_pred == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print("Adversarial accuracy: %.2f%%" % (acc * 100))

        # TODO: use other measures to compute the results
        #print(clever_t(classifier=classifier, x=x_test, target_class=y_test, batch_size=100, nb_batches=10, radius=0.2, norm=2))

        return x_test_adv, x_test_adv_pred

    def save_model(self, classifier, model_name):
        """
        Saves the trained model and adds the current datetime to the filename.
        Example of saved model: `trained_models/2019-05-20/baseline.h5`

        :param classifier: trained classifier
        :param model_name: name of the model
        """
        if self.trained:
            classifier.save(filename=model_name+".h5",  # "_"+time.strftime('%H:%M')+".h5",
                            path=TRAINED_MODELS+time.strftime('%Y-%m-%d')+"/")

    def load_classifier(self, relative_path):
        """
        Loads a pretrained classifier.
        :param relative_path: is the relative path w.r.t. trained_models folder, `2019-05-20/baseline.h5` in the example
        from save_model()
        returns: trained classifier
        """
        # load a trained model
        trained_model = load_model(TRAINED_MODELS+relative_path)
        classifier = KerasClassifier((MIN, MAX), trained_model, use_logits=False)
        return classifier

    def adversarial_train(self, classifier, x_train, y_train, x_test, y_test, batch_size, epochs, method='fgsm'):

        # generate adversarial examples on train and test sets
        x_train_adv = self._generate_adversaries(classifier, x_train, method=method)
        x_test_adv = self._generate_adversaries(classifier, x_test, method=method)

        # Data augmentation: expand the training set with the adversarial samples
        x_train_ext = np.append(x_train, x_train_adv, axis=0)
        y_train_ext = np.append(y_train, y_train, axis=0)

        # Retrain the CNN on the extended dataset
        classifier = self.train(x_train_ext, y_train_ext, batch_size=batch_size, epochs=epochs)

        # Evaluate the adversarially trained classifier on the test set
        self.evaluate_test(classifier, x_test_adv, y_test)

        return classifier

