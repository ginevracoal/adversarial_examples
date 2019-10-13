# -*- coding: utf-8 -*-

from keras.models import load_model
from art.classifiers import KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier as sklKerasClassifier
from art.attacks import FastGradientMethod, DeepFool, VirtualAdversarialMethod, CarliniL2Method,\
    ProjectedGradientDescent, NewtonFool, CarliniLInfMethod, BoundaryAttack, SpatialTransformation, ZooAttack
from sklearn.metrics import classification_report
from utils import *
import time
from art.utils import to_categorical

############
# defaults #
############

TRAINED_MODELS = "../trained_models/"
DATA_PATH = "../data/"
RESULTS = "../results/"
MIN = 0
MAX = 255
DATASETS = "mnist, cifar"
ATTACKS = "None, fgsm, pgd, deepfool, carlini"

BASECLASS = "art" # "skl" # todo: implement skl version


class AdversarialClassifier(object):
    """
    Keras Classifier base class
    """

    def __init__(self, input_shape, num_classes, data_format, dataset_name, test):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.data_format = data_format
        self.model = self._set_model()  # this is needed for training with skl wrapper
        self.batch_size, self.epochs = self._set_training_params(dataset_name=dataset_name, test=test).values()
        self.trained = False

    def _set_training_params(self, dataset_name, test):
        """
        Defines training parameters
        :param dataset_name:
        :param test: if True only takes the first 100 samples
        :return: batch_size, epochs
        """
        if test:
            return {'batch_size':10,'epochs':1}
        else:
            if dataset_name == "mnist":
                return {'batch_size':128,'epochs':12}
            elif dataset_name == "cifar":
                return {'batch_size':128,'epochs':800}

    def _set_model(self):
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

        if BASECLASS == "art":
            classifier = KerasClassifier((MIN, MAX), model=self.model, use_logits=False)
            self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                   metrics=['accuracy'])
            start_time = time.time()
            classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs)

        elif BASECLASS == "skl":

            mini_batch = 20
            early_stopping = keras.callbacks.EarlyStopping(monitor='loss', verbose=1)
            self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                               metrics=['accuracy'])
            start_time = time.time()
            classifier = self.model.fit(x_train, y_train, epochs=epochs, batch_size=mini_batch, callbacks=[early_stopping])

        else:
            raise AssertionError("\nChoose a baseclass: 'art','skl'.")

        print("\nTraining time: --- %s seconds ---" % (time.time() - start_time))
        self.trained = True
        return classifier

    def _get_norm(self, method):
        """ Returns the norm used for computing perturbations on the given method. """
        if method == "deepfool":
            return 2
        else:
            return np.inf

    def generate_adversaries(self, classifier, x, y, test, method, dataset_name, eps):
        """
        Generates adversaries on the input data x using a given method.

        :param classifier: trained classifier
        :param x: input data
        :param method: art.attack method
        :param adversaries_path: path of saved pickle data
        :return: adversarially perturbed data
        """

        print("\nGenerating adversaries with", method, "method on", dataset_name)
        x_adv = None
        if method == 'fgsm':
            attacker = FastGradientMethod(classifier, eps=eps)
            x_adv = attacker.generate(x)
        elif method == 'pgd':
            attacker = ProjectedGradientDescent(classifier, eps=eps)
            x_adv = attacker.generate(x=x, y=y)
        elif method == 'deepfool':
            attacker = DeepFool(classifier, epsilon=0.5)
            x_adv = attacker.generate(x)
        elif method == 'virtual':
            attacker = VirtualAdversarialMethod(classifier)
            x_adv = attacker.generate(x)
        elif method == 'carlini':
            attacker = CarliniLInfMethod(classifier, targeted=False, eps=eps)
            x_adv = attacker.generate(x=x, y=y)
        elif method == 'newtonfool':
            attacker = NewtonFool(classifier)
            x_adv = attacker.generate(x=x)
        elif method == 'boundary':
            attacker = BoundaryAttack(classifier, targeted=True, max_iter=500, delta=0.05, epsilon=0.5)
            y=np.random.permutation(y)
            x_adv = attacker.generate(x=x, y=y)
        elif method == 'spatial':
            attacker = SpatialTransformation(classifier, max_translation=3.0,num_translations=5, max_rotation=8.0,
                                             num_rotations=3)
            x_adv = attacker.generate(x=x, y=y)
        elif method == 'zoo':
            attacker = ZooAttack(classifier)
            x_adv = attacker.generate(x=x, y=y)

        print("Distance from perturbations: ", compute_distances(x, x_adv, ord=self._get_norm(method)))
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
        if BASECLASS == "art":
            predictions = classifier.predict(x)
        elif BASECLASS == "skl":
            predictions = classifier.model.predict(x)
        else:
            raise AssertionError("\nChoose a baseclass: 'art','skl'.")

        # print(predictions[:, 0])
        return predictions

    def evaluate(self, classifier, x, y):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param classifier: trained classifier
        :param x: test data
        :param y: test labels
        :return: predictions
        """
        print("\n===== Classifier evaluation =====")
        print("\nTesting infos:\nx.shape = ", x.shape, "\ny.shape = ", y.shape, "\n")

        y_pred = np.argmax(self.predict(classifier, x), axis=1)
        y_true = np.argmax(y, axis=1)
        correct_preds = np.sum(y_pred == np.argmax(y, axis=1))

        print("Correctly classified: {}".format(correct_preds))
        print("Incorrectly classified: {}".format(len(x) - correct_preds))

        # print(y_test_pred, y_test_true)
        acc = np.sum(y_pred == y_true) / y.shape[0]
        print("Test accuracy: %.2f%%" % (acc * 100))

        # classification report over single classes
        print(classification_report(y_true, y_pred, labels=range(self.num_classes)))

        return y_pred

    def save_classifier(self, classifier, model_name):
        """
        Saves the trained model and adds the current datetime to the filename.
        Example of saved model: `trained_models/2019-05-20/baseline.h5`

        :param classifier: trained classifier
        :param model_name: name of the model
        """
        if self.trained:
            classifier.save(filename=model_name+".h5", path=RESULTS + time.strftime('%Y-%m-%d') + "/")

    def load_classifier(self, path):
        """
        Loads a pre-trained classifier.
        :param relative_path: relative path w.r.t. trained_models folder
        returns: trained classifier
        """
        print("\nLoading model:", str(path))
        # load a trained model
        trained_model = load_model(path)
        classifier = KerasClassifier((MIN, MAX), trained_model, use_logits=False)
        return classifier

    def adversarial_train(self, classifier, dataset_name, x_train, y_train, method, eps, batch_size, epochs, test=False):
        """
        Performs adversarial training on the given classifier using an attack method.
        :param classifier: trained classifier
        :param method: attack method
        :return: robust classifier
        """

        start_time = time.time()
        print("\n===== Adversarial training =====")
        # generate adversarial examples on train and test sets
        x_train_adv = self.generate_adversaries(classifier, x_train, y_train, method=method,
                                                dataset_name=dataset_name, test=test, eps=eps)

        # Data augmentation: expand the training set with the adversarial samples
        x_train_ext = np.append(x_train, x_train_adv, axis=0)
        y_train_ext = np.append(y_train, y_train, axis=0)

        # Retrain the CNN on the extended dataset
        robust_classifier = self.train(x_train_ext, y_train_ext, batch_size=batch_size, epochs=epochs)
        print("\nAdversarial training time: --- %s seconds ---" % (time.time() - start_time))

        return robust_classifier

    @staticmethod
    def save_adversaries(data, dataset_name, attack, eps):
        """
        Save adversarially augmented test set.
        :param data: test set
        :param dataset_name:
        :param attack:
        :param eps:
        :return:
        """
        if attack == "deepfool":
            save_to_pickle(data=data, filename=dataset_name + "_x_test_" + attack + ".pkl")
        else:
            save_to_pickle(data=data, filename=dataset_name + "_x_test_" + attack + "_" + str(eps) + ".pkl")

    @staticmethod
    def load_adversaries(dataset_name, attack, eps, test):
        print("\nLoading adversaries generated with", attack, "method on", dataset_name)
        if attack == "deepfool":
            x_test_adv = load_from_pickle(path=DATA_PATH + dataset_name + "_x_test_" + attack + ".pkl", test=test)
        else:
            x_test_adv = load_from_pickle(path=DATA_PATH + dataset_name + "_x_test_" + attack + "_" +
                                               str(eps) + ".pkl", test=test)

        return x_test_adv

