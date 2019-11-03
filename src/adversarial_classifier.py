# -*- coding: utf-8 -*-

from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier as sklKerasClassifier
from art.classifiers import KerasClassifier as artKerasClassifier
from art.attacks import FastGradientMethod, DeepFool, VirtualAdversarialMethod, \
    ProjectedGradientDescent, NewtonFool, CarliniLInfMethod #, BoundaryAttack, SpatialTransformation, ZooAttack
from sklearn.metrics import classification_report
from utils import *
import time
import tensorflow as tf
from keras.layers import Input
from tensorflow.python.client import device_lib

############
# defaults #
############

TRAINED_MODELS = "../trained_models/"
DATA_PATH = "../data/"
RESULTS = "../results/"
DATASETS = "mnist, cifar"
ATTACKS = "None, fgsm, pgd, deepfool, carlini"
MINIBATCH = 20


class AdversarialClassifier(sklKerasClassifier):
    """
    Adversarial Classifier base class
    """

    def __init__(self, input_shape, num_classes, data_format, dataset_name, test):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.data_format = data_format
        self.dataset_name = dataset_name
        self.test = test
        self.model = self._set_model()
        self.batch_size, self.epochs = self._set_training_params(test=test).values()
        super(AdversarialClassifier, self).__init__(build_fn=self.model, batch_size=self.batch_size, epochs=self.epochs)
        self.classes_ = self._set_classes()
        self.model_name = None
        self.trained = False
        os.makedirs(os.path.dirname(RESULTS + time.strftime('%Y-%m-%d') + "/"), exist_ok=True)

    # todo: docstrings
    @staticmethod
    def _set_training_params(test):
        raise NotImplementedError

    def _get_logits(self, inputs):
        raise NotImplementedError

    @staticmethod
    def _set_session():
        """ Initialize tf session """
        print("check cuda: ", tf.test.is_built_with_cuda())
        print("check gpu: ", tf.test.is_gpu_available())
        print(device_lib.list_local_devices())
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)

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
            early_stopping = keras.callbacks.EarlyStopping(monitor='loss', verbose=1)
            self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                               metrics=['accuracy'])
            start_time = time.time()
            self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=mini_batch, callbacks=[early_stopping])

            print("\nTraining time: --- %s seconds ---" % (time.time() - start_time))
            self.trained = True
            return self

    def evaluate(self, x, y):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param x: test data
        :param y: test labels
        :return: predictions
        """
        if self.trained:
            y_true = np.argmax(y, axis=1)
            y_pred = self.predict(x)  # or np.argmax(self.model.predict(x), axis=1)
            nb_correct_adv_pred = np.sum(y_pred == y_true)

            print("\nTest data.")
            print("Correctly classified: {}".format(nb_correct_adv_pred))
            print("Incorrectly classified: {}".format(len(x) - nb_correct_adv_pred))

            acc = nb_correct_adv_pred / y.shape[0]
            print("Accuracy: %.2f%%" % (acc * 100))
            print(classification_report(y_true, y_pred, labels=list(range(self.num_classes))))
        else:
            raise AttributeError("Train your classifier before the evaluation.")

    @staticmethod
    def _get_norm(attack):
        """ Returns the norm used for computing perturbations on the given method. """
        if attack == "deepfool":
            return 2
        else:
            return np.inf

    def generate_adversaries(self, x, y, attack, eps=0.5):
        """
        Generates adversaries on the input data x using a given attack method.

        :param classifier: trained classifier
        :param x: input data
        :param attack: art.attack method
        :return: adversarially perturbed data
        """
        if self.trained:
            classifier = artKerasClassifier((0, 255), self.model, use_logits=False)
        else:
            raise AttributeError("Train your classifier first.")

        # classifier._loss = self.loss_wrapper(tf.convert_to_tensor(x), None)
        # classifier.custom_loss = self.loss_wrapper(tf.convert_to_tensor(x), None)
        print("\nGenerating adversaries with", attack, "method on", self.dataset_name)
        x_adv = None
        if attack == 'fgsm':
            attacker = FastGradientMethod(classifier, eps=eps)
            x_adv = attacker.generate(x=x)
        elif attack == 'deepfool':
            attacker = DeepFool(classifier)
            x_adv = attacker.generate(x)
        elif attack == 'virtual':
            attacker = VirtualAdversarialMethod(classifier)
            x_adv = attacker.generate(x)
        elif attack == 'carlini':
            attacker = CarliniLInfMethod(classifier, targeted=False)
            x_adv = attacker.generate(x=x, y=y)
        elif attack == 'pgd':
            attacker = ProjectedGradientDescent(classifier)
            x_adv = attacker.generate(x=x)
        elif attack == 'newtonfool':
            attacker = NewtonFool(classifier)
            x_adv = attacker.generate(x=x)
        # elif attack == 'boundary':
        #     attacker = BoundaryAttack(classifier, targeted=True, max_iter=500, delta=0.05, epsilon=0.5)
        #     y=np.random.permutation(y)
        #     x_adv = attacker.generate(x=x, y=y)
        # elif attack == 'spatial':
        #     attacker = SpatialTransformation(classifier, max_translation=3.0,num_translations=5, max_rotation=8.0,
        #                                      num_rotations=3)
        #     x_adv = attacker.generate(x=x, y=y)
        # elif attack == 'zoo':
        #     attacker = ZooAttack(classifier)
        #     x_adv = attacker.generate(x=x, y=y)

        print("Distance from perturbations: ", compute_distances(x, x_adv, ord=self._get_norm(attack)))

        if self.test:
            return x_adv[:TEST_SIZE]
        else:
            return x_adv

    @staticmethod
    def save_adversaries(data, dataset_name, attack, eps):
        """
        #todo docstring
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

    def save_classifier(self, relative_path=RESULTS + time.strftime('%Y-%m-%d') + "/"):
        """
        Saves the trained model and adds the current datetime to the filepath.
        :relative_path: path of folder containing the trained model
        """
        self.model.save_weights(relative_path+self.model_name)

    def load_classifier(self, relative_path):
        """
        Loads a pre-trained classifier.
        :relative_path: path of folder containing the trained model
        returns: trained classifier
        """
        self.model.load_weights(relative_path + self.model_name + ".h5")
        self.trained = True
        return self

