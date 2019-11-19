# -*- coding: utf-8 -*-

from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier as sklKerasClassifier
from art.classifiers import KerasClassifier as artKerasClassifier
from art.attacks import FastGradientMethod, DeepFool, VirtualAdversarialMethod, \
    ProjectedGradientDescent, NewtonFool, CarliniL2Method, CarliniLInfMethod, BoundaryAttack, SpatialTransformation, ZooAttack
from sklearn.metrics import classification_report
from utils import *
import time
import tensorflow as tf
from keras.layers import Input
from tensorflow.python.client import device_lib
from keras.models import load_model
import random
from art.utils import master_seed

############
# defaults #
############

MINIBATCH = 20
TRAINED_MODELS = "../trained_models/"
DATA_PATH = "../data/"
RESULTS = "../results/"+str(time.strftime('%Y-%m-%d'))+"/"
DATASETS = "mnist, cifar"
ATTACKS = "None, fgsm, pgd, deepfool, carlini"


class AdversarialClassifier(sklKerasClassifier):
    """
    Adversarial Classifier base class
    """

    def __init__(self, input_shape, num_classes, data_format, dataset_name, test, epochs=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.data_format = data_format
        self.dataset_name = dataset_name
        self.test = test
        self.model = self._set_model()
        self.batch_size, self.epochs = self._set_training_params(test=test, epochs=epochs).values()
        super(AdversarialClassifier, self).__init__(build_fn=self.model, batch_size=self.batch_size, epochs=self.epochs)
        self.classes_ = self._set_classes()
        self.folder, self.filename = self._set_model_path().values()
        self.trained = False

    # todo: docstrings
    def _set_model_path(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _set_training_params(test, epochs):
        raise NotImplementedError

    def _get_logits(self, inputs):
        raise NotImplementedError

    @staticmethod
    def _set_session(device):
        """ Initialize tf session """
        print(device_lib.list_local_devices())

        if device == "gpu":
            print("check cuda: ", tf.test.is_built_with_cuda())
            print("check gpu: ", tf.test.is_gpu_available())
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # sess = tf.Session(config=config)
            from keras.backend.tensorflow_backend import set_session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
            config.log_device_placement = True  # to log device placement (on which device the operation ran)
            sess = tf.Session(config=config)
            set_session(sess)  # set this TensorFlow session as the default session for Keras
            return sess
        elif device == "cpu":
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            # sess.run(tf.global_variables_initializer())
            return sess

    def _set_model(self):
        """
        defines the layers structure for the classifier
        :return: model
        """
        inputs = Input(shape=self.input_shape)
        outputs = self._get_logits(inputs=inputs)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def _set_classes():
        """ Setting classes_ attribute for sklearn KerasClassifier class """
        return np.array(np.arange(10))

    def _set_device_name(self, device):
        if device == "gpu":
            return "/device:GPU:0"
            # device_name = "/job:localhost/replica:0/task:0/device:XLA_GPU:0')"
        elif device == "cpu":
            return "/CPU:0"
        else:
            raise AssertionError("Wrong device name.")

    def train(self, x_train, y_train, device):
        print("\nTraining infos:\nbatch_size = ", self.batch_size, "\nepochs = ", self.epochs,
              "\nx_train.shape = ", x_train.shape, "\ny_train.shape = ", y_train.shape, "\n")

        device_name = self._set_device_name(device)
        with tf.device(device_name):
            mini_batch = MINIBATCH
            self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                               metrics=['accuracy'])
            start_time = time.time()
            if self.epochs == None:
                es = keras.callbacks.EarlyStopping(monitor='loss', verbose=1)
                self.model.fit(x_train, y_train, epochs=50, batch_size=mini_batch, callbacks=[es], shuffle=True)
            else:
                self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=mini_batch, shuffle=True)
            print("\nTraining time: --- %s seconds ---" % (time.time() - start_time))
            self.trained = True
            return self

    def predict(self, x, **kwargs):
        return self.model.predict(x)
        # return np.argmax(self.model.predict(x), axis=1)

    def evaluate(self, x, y):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param x: test data
        :param y: test labels
        :return: predictions
        """
        if self.trained:
            y_true = np.argmax(y, axis=1)
            y_pred = np.argmax(self.predict(x), axis=1)
            nb_correct_adv_pred = np.sum(y_pred == y_true)

            print("Correctly classified: {}".format(nb_correct_adv_pred))
            print("Incorrectly classified: {}".format(len(x) - nb_correct_adv_pred))

            acc = nb_correct_adv_pred / y.shape[0]
            print("Accuracy: %.2f%%" % (acc * 100))
            # print(classification_report(y_true, y_pred, labels=list(range(self.num_classes))))
        else:
            raise AttributeError("Train your classifier before the evaluation.")

    @staticmethod
    def _get_norm(attack):
        """ Returns the norm used for computing perturbations on the given method. """
        return np.inf

    def _get_attack_eps(self, dataset_name, attack):
        if dataset_name == "mnist":
            eps = {'fgsm': 0.3, 'pgd': 0.3, 'carlini': 0.8, 'deepfool': None, 'newtonfool':None}
        elif dataset_name == "cifar":
            eps = {'fgsm': 0.3, 'pgd': 0.3, 'carlini': 0.5, 'deepfool': None, 'newtonfool':None}
        else:
            raise ValueError("Wrong dataset name.")
        return eps[attack]

    def generate_adversaries(self, x, y, attack, eps=None, seed=0):
        """
        Generates adversaries on the input data x using a given attack method.

        :param classifier: trained classifier
        :param x: input data
        :param attack: art.attack method
        :return: adversarially perturbed data
        """

        if self.trained:
            classifier = artKerasClassifier(clip_values=(0,1), model=self.model)
            master_seed(seed)
            # random.seed(seed)
        else:
            raise AttributeError("Train your classifier first.")

        if eps is None:
            eps = self._get_attack_eps(dataset_name=self.dataset_name, attack=attack)

        # classifier._loss = self.loss_wrapper(tf.convert_to_tensor(x), None)
        # classifier.custom_loss = self.loss_wrapper(tf.convert_to_tensor(x), None)
        print("\nGenerating adversaries with", attack, "method on", self.dataset_name)
        x_adv = None
        if attack == 'fgsm':
            attacker = FastGradientMethod(classifier, eps=eps)
            x_adv = attacker.generate(x=x)
        elif attack == 'deepfool':
            attacker = DeepFool(classifier, nb_grads=5)
            x_adv = attacker.generate(x)
        elif attack == 'virtual':
            attacker = VirtualAdversarialMethod(classifier)
            x_adv = attacker.generate(x)
        elif attack == 'carlini':
            attacker = CarliniLInfMethod(classifier, targeted=False, eps=eps)
            x_adv = attacker.generate(x=x)
        elif attack == 'pgd':
            attacker = ProjectedGradientDescent(classifier, eps=eps)
            x_adv = attacker.generate(x=x)
        elif attack == 'newtonfool':
            attacker = NewtonFool(classifier, eta=0.3)
            x_adv = attacker.generate(x=x)
        elif attack == 'boundary':
            attacker = BoundaryAttack(classifier, targeted=True, max_iter=500, delta=0.05, epsilon=eps)
            y=np.random.permutation(y)
            x_adv = attacker.generate(x=x, y=y)
        elif attack == 'spatial':
            attacker = SpatialTransformation(classifier, max_translation=3.0,num_translations=5, max_rotation=8.0,
                                             num_rotations=3)
            x_adv = attacker.generate(x=x, y=y)
        elif attack == 'zoo':
            attacker = ZooAttack(classifier)
            x_adv = attacker.generate(x=x, y=y)

        print("Distance from perturbations: ", compute_distances(x, x_adv, ord=self._get_norm(attack)))

        if self.test:
            return x_adv[:TEST_SIZE]
        else:
            return x_adv

    def save_adversaries(self, data, attack, eps, seed=0):
        """
        Save adversarially augmented test set.
        :param data: test set
        :param dataset_name:
        :param attack:
        :param eps:
        :return:
        """
        if eps:
            save_to_pickle(data=data, relative_path=RESULTS,
                           filename=self.dataset_name + "_x_test_" + attack + "_" + str(eps) + "_" + str(seed) + ".pkl")
        else:
            save_to_pickle(data=data, relative_path=RESULTS,
                           filename=self.dataset_name + "_x_test_" + attack + "_" + str(seed) + ".pkl")

    def load_adversaries(self, attack, seed=0, eps=None):
        if eps:
            path = DATA_PATH + self.dataset_name + "_x_test_" + attack + "_" + str(eps) + "_" + str(seed) + ".pkl"
        else:
            eps = self._get_attack_eps(dataset_name=self.dataset_name, attack=attack)
            if eps is None:
                path = DATA_PATH + self.dataset_name + "_x_test_" + attack + "_" + str(seed) + ".pkl"
            else:
                path = DATA_PATH + self.dataset_name + "_x_test_" + attack + "_" + str(eps) + "_" + str(seed) + ".pkl"

        x_test_adv = load_from_pickle(path=path, test=self.test)
        return x_test_adv

    def save_classifier(self, relative_path, folder=None, filename=None):
        """
        Saves the trained model and adds the current datetime to the filepath.
        :relative_path: path of folder containing the trained model
        """
        if folder is None:
            folder = self.folder
        if filename is None:
            filename = self.filename
        os.makedirs(os.path.dirname(relative_path + folder), exist_ok=True)
        filepath = relative_path + folder + filename + ".h5"
        print("\nSaving classifier: ", filepath)
        self.model.save(filepath)

    def load_classifier(self, relative_path, folder=None, filename=None):
        """
        Loads a pre-trained classifier.
        :relative_path: path of folder containing the trained model
        returns: trained classifier
        """
        if folder is None:
            folder = self.folder
        if filename is None:
            filename = self.filename
        print("\nLoading model: ", relative_path + folder + filename + ".h5")
        self.model = load_model(relative_path + folder + filename + ".h5")
        self.trained = True
        return self

