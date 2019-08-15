# -*- coding: utf-8 -*-

from keras.models import load_model
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod, DeepFool, VirtualAdversarialMethod, CarliniL2Method,\
    ProjectedGradientDescent, NewtonFool, CarliniLInfMethod
from sklearn.metrics import classification_report
from utils import *
import time


############
# defaults #
############

TRAINED_MODELS = "../trained_models/"
DATA_PATH = "../data/"
RESULTS = "../results/"
MIN = 0
MAX = 255
DATASETS = "mnist, cifar"
ATTACKS = "None, fgsm, pgd, deepfool, carlini_linf"


class AdversarialClassifier(object):
    """
    Keras Classifier base class
    """

    def __init__(self, input_shape, num_classes, data_format):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.data_format = data_format
        self.model = self._set_layers()
        self.trained = False

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

        start_time = time.time()
        classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs)
        print("\nTraining time: --- %s seconds ---" % (time.time() - start_time))

        self.trained = True
        return classifier

    def _generate_adversaries(self, classifier, x, y, test, method, dataset_name, adversaries_path=None):
        """
        Generates adversaries on the input data x using a given method or loads saved data if available.

        :param classifier: trained classifier
        :param x: input data
        :param method: art.attack method
        :param adversaries_path: path of saved pickle data
        :return: adversarially perturbed data
        """

        if adversaries_path is None:
            print("\nGenerating adversaries with", method, "method on", dataset_name)
            x_adv = None
            if method == 'fgsm':
                attacker = FastGradientMethod(classifier, eps=0.5)
                x_adv = attacker.generate(x)
            elif method == 'deepfool':
                attacker = DeepFool(classifier)
                x_adv = attacker.generate(x)
            elif method == 'virtual':
                attacker = VirtualAdversarialMethod(classifier)
                x_adv = attacker.generate(x)
            elif method == 'carlini_l2':
                attacker = CarliniL2Method(classifier, targeted=False)
                x_adv = attacker.generate(x=x, y=y)
            elif method == 'carlini_linf':
                attacker = CarliniLInfMethod(classifier, targeted=False)
                x_adv = attacker.generate(x=x, y=y)
            elif method == 'pgd':
                attacker = ProjectedGradientDescent(classifier)
                x_adv = attacker.generate(x=x)
            elif method == 'newtonfool':
                attacker = NewtonFool(classifier)
                x_adv = attacker.generate(x=x)

        else:
            print("\nLoading adversaries generated with", method, "method on", dataset_name)
            x_adv = load_from_pickle(path=adversaries_path, test=test)  # [0]

        if test:
            return x_adv[:TEST_SIZE]
        else:
            return x_adv

    def predict(self, classifier, x, *args, **kwargs):
        """
        This method is needed for calling the method predict on other objects than keras classifiers in the derived
        classes.
        :param classifier: trained classifier
        :param x: input data
        :return: predictions
        """
        predictions = classifier.predict(x)
        # print(predictions[:, 0])
        return predictions

    def evaluate_test(self, classifier, x_test, y_test):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        :return: x_test predictions
        """
        print("\n===== Test set evaluation =====")
        print("\nTesting infos:\nx_test.shape = ", x_test.shape, "\ny_test.shape = ", y_test.shape, "\n")

        y_test_pred = np.argmax(self.predict(classifier, x_test), axis=1)
        y_test_true = np.argmax(y_test, axis=1)
        correct_preds = np.sum(y_test_pred == np.argmax(y_test, axis=1))

        print("Correctly classified: {}".format(correct_preds))
        print("Incorrectly classified: {}".format(len(x_test) - correct_preds))

        acc = np.sum(y_test_pred == y_test_true) / y_test.shape[0]
        print("Test accuracy: %.2f%%" % (acc * 100))

        # classification report over single classes
        print(classification_report(np.argmax(y_test, axis=1), y_test_pred, labels=range(self.num_classes)))

        return y_test_pred

    def evaluate_adversaries(self, classifier, x_test, y_test, method, dataset_name, adversaries_path=None, test=False):
        """
        Evaluates the trained model against FGSM and prints the number of misclassifications.
        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        :param method: art.attack method
        :param adversaries_path: path of saved pickle data
        :param test: if true only takes TEST_SIZE samples
        :return:
        x_test_pred: test set predictions
        x_test_adv: adversarial perturbations of test data
        y_test_adv: adversarial test set predictions
        """
        print("\n===== Adversarial evaluation =====")

        # generate adversaries on the test set
        x_test_adv = self._generate_adversaries(classifier, x_test, y_test, method=method, dataset_name=dataset_name,
                                                adversaries_path=adversaries_path, test=test)
        # debug
        print(len(x_test), len(x_test_adv))

        # evaluate the performance on the adversarial test set
        y_test_adv = np.argmax(self.predict(classifier, x_test_adv), axis=1)
        y_test_true = np.argmax(y_test, axis=1)
        nb_correct_adv_pred = np.sum(y_test_adv == y_test_true)

        print("\nAdversarial test data.")
        print("Correctly classified: {}".format(nb_correct_adv_pred))
        print("Incorrectly classified: {}".format(len(x_test_adv) - nb_correct_adv_pred))

        acc = nb_correct_adv_pred / y_test.shape[0]
        print("Adversarial accuracy: %.2f%%" % (acc * 100))

        # classification report
        print(classification_report(np.argmax(y_test, axis=1), y_test_adv, labels=range(self.num_classes)))

        return x_test_adv, y_test_adv

    def save_model(self, classifier, model_name):
        """
        Saves the trained model and adds the current datetime to the filename.
        Example of saved model: `trained_models/2019-05-20/baseline.h5`

        :param classifier: trained classifier
        :param model_name: name of the model
        """
        if self.trained:
            classifier.save(filename=model_name+".h5", path=RESULTS + time.strftime('%Y-%m-%d') + "/")

    def load_classifier(self, relative_path):
        """
        Loads a pretrained classifier.
        :param relative_path: is the relative path w.r.t. trained_models folder, `2019-05-20/baseline.h5` in the example
        from save_model()
        returns: trained classifier
        """
        print("\nLoading model:", str(relative_path))
        # load a trained model
        trained_model = load_model(relative_path)
        classifier = KerasClassifier((MIN, MAX), trained_model, use_logits=False)
        return classifier

    def adversarial_train(self, classifier, x_train, y_train, batch_size, epochs, method, dataset_name, test=False):
        """
        Performs adversarial training on the given classifier using an attack method.
        :param classifier: trained classifier
        :param method: attack method
        :return: robust classifier
        """

        start_time = time.time()
        print("\n===== Adversarial training =====")
        # generate adversarial examples on train and test sets
        x_train_adv = self._generate_adversaries(classifier, x_train, y_train, method=method,
                                                 dataset_name=dataset_name, test=test)

        # Data augmentation: expand the training set with the adversarial samples
        x_train_ext = np.append(x_train, x_train_adv, axis=0)
        y_train_ext = np.append(y_train, y_train, axis=0)

        # Retrain the CNN on the extended dataset
        robust_classifier = self.train(x_train_ext, y_train_ext, batch_size=batch_size, epochs=epochs)
        print("\nAdversarial training time: --- %s seconds ---" % (time.time() - start_time))

        # todo: remove next lines... they have nothing do to with adversarial training
        # Evaluate the adversarially trained classifier on the original + adversarial test sets
        #print("\nEvaluating on original test set:")
        #self.evaluate_test(robust_classifier, x_test, y_test)

        #print("\nEvaluating on adversarial test set generated from the original classifier:")
        #self.evaluate_adversaries(robust_classifier, x_test, y_test, method=method, test=test, dataset_name=dataset_name,
        #                           adversaries_path="../data/"+dataset_name+"_x_test_" + method + ".pkl")
        # same as
        # self.evaluate_test(robust_classifier, x_test_adv, y_test)

        # save_to_pickle(data=x_train_adv, filename="x_train_adv_" + method + ".pkl")
        return robust_classifier

