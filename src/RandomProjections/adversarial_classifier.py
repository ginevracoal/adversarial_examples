# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
from directories import *

from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier as sklKerasClassifier
from sklearn.metrics import classification_report
from utils import *
from utils import _set_session
import time
import tensorflow as tf
from keras.layers import Input
from tensorflow.python.client import device_lib
from keras.models import load_model
import random
import warnings

############
# defaults #
############

MINIBATCH = 20
DATASETS = "mnist, cifar"
ATTACKS = "None, fgsm, pgd, deepfool, carlini, newtonfool, virtual"


class AdversarialClassifier(sklKerasClassifier):
    """
    Adversarial Classifier base class
    """

    def __init__(self, input_shape, num_classes, data_format, dataset_name, test, library, epochs=None):
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
        self.library = library  # art, cleverhans

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
        # print(device_lib.list_local_devices())

        if device == "gpu":
            n_jobs = 1
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
            config.gpu_options.per_process_gpu_memory_fraction = 1 / n_jobs
            sess = tf.compat.v1.Session(config=config)
            keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
            sess.run(tf.global_variables_initializer())
            return sess
        elif device == "cpu":
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            sess.run(tf.global_variables_initializer())
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
        elif device == "cpu":
            return "/CPU:0"
        else:
            raise AssertionError("Wrong device name.")

    def set_optimizer(self):
        return keras.optimizers.Adadelta()

    def train(self, x_train, y_train, device):
        print("\nTraining infos:\nbatch_size = ", self.batch_size, "\nepochs = ", self.epochs,
              "\nx_train.shape = ", x_train.shape, "\ny_train.shape = ", y_train.shape, "\n")

        device_name = self._set_device_name(device)
        with tf.device(device_name):
            mini_batch = MINIBATCH
            optimizer = self.set_optimizer()
            self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

            # callbacks
            callbacks = []
            tensorboard = keras.callbacks.TensorBoard(log_dir='../tensorboard/', histogram_freq=0, write_graph=True,
                                                      write_images=True)
            es = keras.callbacks.EarlyStopping(monitor='loss', verbose=1)
            if self.epochs == None:
                epochs = 50
                callbacks.append(es)
            else:
                epochs = self.epochs
            if self.test == False:
                callbacks.append(tensorboard)

            # training
            start_time = time.time()
            self.model.fit(x_train, y_train, epochs=epochs, batch_size=mini_batch, callbacks=callbacks,
                               shuffle=True, validation_split=0.2)
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
            classification_prob = self.predict(x)
            y_true = np.argmax(y, axis=1)
            y_pred = np.argmax(classification_prob, axis=1)
            nb_correct_adv_pred = np.sum(y_pred == y_true)

            print("Correctly classified: {}".format(nb_correct_adv_pred))
            print("Incorrectly classified: {}".format(len(x) - nb_correct_adv_pred))

            acc = nb_correct_adv_pred / y.shape[0]
            print("Accuracy: %.2f%%" % (acc * 100))
            # print(classification_report(y_true, y_pred, labels=list(range(self.num_classes))))
            return classification_prob, y_true, y_pred
        else:
            raise AttributeError("Train your classifier before the evaluation.")

    @staticmethod
    def _get_norm(attack):
        """ Returns the norm used for computing perturbations on the given method. """
        return np.inf

    def generate_adversaries(self, x, y, attack, eps=None, seed=0, device="cpu"):
        """
        Generates adversaries on the input data x using a given attack method.

        :param classifier: trained classifier
        :param x: input data
        :param attack: art.attack method
        :return: adversarially perturbed data
        """
        random.seed(seed)
        def batch_generate(attacker, x, batches=10):
            x_batches = np.split(x, batches)
            x_adv = []
            for idx, x_batch in enumerate(x_batches):
                x_adv.append(attacker.generate_np(x_val=x_batch))
            x_adv = np.vstack(x_adv)
            return x_adv

        x_adv = None

        if self.trained:
            print("\nGenerating adversaries with", attack, "method on", self.dataset_name)
            with warnings.catch_warnings():
                if self.library == "art":
                    import art.attacks
                    from art.classifiers import KerasClassifier as artKerasClassifier
                    from art.utils import master_seed

                    classifier = artKerasClassifier(clip_values=(0,255), model=self.model)
                    master_seed(seed)

                    # classifier._loss = self.loss_wrapper(tf.convert_to_tensor(x), None)
                    # classifier.custom_loss = self.loss_wrapper(tf.convert_to_tensor(x), None)
                    if attack == 'fgsm':
                        attacker = art.attacks.FastGradientMethod(classifier, eps=eps)
                        x_adv = attacker.generate(x=x)
                    elif attack == 'deepfool':
                        attacker = art.attacks.DeepFool(classifier, nb_grads=5)
                        x_adv = attacker.generate(x)
                    elif attack == 'virtual':
                        attacker = art.attacks.VirtualAdversarialMethod(classifier)
                        x_adv = attacker.generate(x)
                    elif attack == 'carlini':
                        attacker = art.attacks.CarliniLInfMethod(classifier, targeted=False, eps=0.5)
                        x_adv = attacker.generate(x=x)
                    elif attack == 'pgd':
                        attacker = art.attacks.ProjectedGradientDescent(classifier, eps=eps)
                        x_adv = attacker.generate(x=x)
                    elif attack == 'newtonfool':
                        attacker = art.attacks.NewtonFool(classifier, eta=0.3)
                        x_adv = attacker.generate(x=x)
                    elif attack == 'boundary':
                        attacker = art.attacks.BoundaryAttack(classifier, targeted=False, max_iter=500, delta=0.05)
                        # y = np.random.permutation(y)
                        x_adv = attacker.generate(x=x)
                    elif attack == 'spatial':
                        attacker = art.attacks.SpatialTransformation(classifier, max_translation=3.0, num_translations=5,
                                                         max_rotation=8.0,
                                                         num_rotations=3)
                        x_adv = attacker.generate(x=x)
                    elif attack == 'zoo':
                        attacker = art.attacks.ZooAttack(classifier)
                        x_adv = attacker.generate(x=x, y=y)
                    else:
                        raise("wrong attack name.")

                elif self.library == "cleverhans":
                    import cleverhans.attacks
                    from cleverhans.utils_keras import KerasModelWrapper

                    session = self._set_session(device=device)
                    classifier = KerasModelWrapper(self.model)

                    if attack == 'fgsm':
                        attacker = cleverhans.attacks.FastGradientMethod(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                    elif attack == 'deepfool':
                        attacker = cleverhans.attacks.DeepFool(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                    elif attack == 'carlini':
                        attacker = cleverhans.attacks.CarliniWagnerL2(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                    elif attack == 'pgd':
                        attacker = cleverhans.attacks.ProjectedGradientDescent(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                    elif attack == 'spatial':
                        attacker = cleverhans.attacks.SpatialTransformationMethod(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                    elif attack == 'virtual':
                        attacker = cleverhans.attacks.VirtualAdversarialMethod(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                    elif attack == 'saliency':
                        attacker = cleverhans.attacks.SaliencyMapMethod(classifier, sess=session)
                        x_adv = batch_generate(attacker, x)
                else:
                    raise ValueError("wrong pkg name.")
        else:
            raise AttributeError("Train your classifier first.")

        print("Distance from perturbations: ", compute_distances(x, x_adv, ord=self._get_norm(attack)))
        return x_adv

    def save_adversaries(self, data, attack, eps=None, seed=0):
        """
        Save adversarially augmented test set.
        :param data: test set
        :param dataset_name:
        :param attack:
        :param eps:
        :return:
        """
        filename = self.dataset_name + "_x_test_" + attack + "_"+str(self.library)+ "_seed=" + str(seed) + ".pkl"
        eps_filename = self.dataset_name + "_x_test_" + attack + "_eps=" + str(eps) + "_"+str(self.library)+ "_seed=" \
                       + str(seed) + ".pkl"

        if eps:
            save_to_pickle(data=data, relative_path=RESULTS, filename=eps_filename)
        else:
            save_to_pickle(data=data, relative_path=RESULTS, filename=filename)

    def load_adversaries(self, relative_path, attack, seed=0, eps=None):
        path = relative_path + self.dataset_name + "_x_test_" + attack + "_"+ str(self.library) + "_seed=" + str(seed) + ".pkl"
        if eps:
            eps_path = relative_path + self.dataset_name + "_x_test_" + attack + "_eps=" + str(eps) +\
                       "_"+str(self.library)+ "_seed=" + str(seed) + ".pkl"
            return load_from_pickle(path=eps_path, test=self.test)
        else:
            return load_from_pickle(path=path, test=self.test)

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

