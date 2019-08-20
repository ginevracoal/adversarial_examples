# -*- coding: utf-8 -*-

from adversarial_classifier import *
from art.classifiers import TFClassifier
import tensorflow as tf
from keras.models import load_model
from utils import *
import sys
import random
from keras.models import Model
from baseline_convnet import  BaselineConvnet
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization
from art.classifiers import KerasClassifier as artKerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier as sklKerasClassifier
from keras.callbacks import Callback
from tensorflow.python.keras.layers import Lambda
import keras.losses

# todo: docstrings!
# todo: unittest


class RandomRegularizer(sklKerasClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, sess, lam, test):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        self.sess=sess
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.data_format = data_format
        self.lam = lam
        self.batch_size, self.epochs = self._set_training_params()
        self.model = self._set_layers()
        #self.epoch = K.variable(value=0)
        self.inputs = Input(shape=self.input_shape)
        super(RandomRegularizer, self).__init__(build_fn=self._set_layers, batch_size=self.batch_size, epochs=self.epochs)

    def _set_training_params(self):
        if self.dataset_name == "mnist":
            if test:
                return 100, 12
            else:
                return 1000, 12
        elif self.dataset_name == "cifar":
            if test:
                return 100, 120
            else:
                return 1000, 120

    def _get_logits(self, inputs):
        if self.dataset_name == "mnist":
            x = Conv2D(32, kernel_size=(3, 3), activation='relu', data_format=self.data_format)(inputs)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            logits = Dense(self.num_classes, activation='softmax')(x)
            return logits

        elif self.dataset_name == "cifar":
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
            x = BatchNormalization()(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.2)(x)
            x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.3)(x)
            x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.4)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            logits = Dense(10, activation='softmax')(x)
            return logits

    def _set_layers(self):
        inputs = Input(shape=self.input_shape)
        logits = self._get_logits(inputs=inputs)
        self.outputs = logits
        model = Model(inputs=inputs, outputs=logits)
        model.compile(loss=self.loss_wrapper(inputs, 0), optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
        # model.summary()
        self.model = model
        return model

    def loss_wrapper(self, inputs, batch):
        def custom_loss(y_true, y_pred):
            return K.binary_crossentropy(y_true, y_pred) + self.lam * self.regularizer(inputs=inputs, batch=batch, y_true=y_true)
        return custom_loss

    def regularizer(self, inputs, batch, y_true, max_nproj=6):
        """
        This regularization term penalizes the expected value of loss gradient, evaluated on random projections of the
        input points.
        :param inputs:
        :param y_true:
        :param max_nproj:
        :return: regularization term for the objective loss function.
        """
        rows, cols, channels = self.input_shape
        flat_images = tf.reshape(inputs, shape=[self.batch_size, rows * cols * channels])

        #n_proj = 3  # random.randint(3, max_proj)
        n_features = rows * cols * channels
        regularization = 0

        size = random.randint(6,20)
        seed = random.randint(1,100)
        n_components = size*size*channels
        print("\nsize =", size, ", seed =", seed)
        projector = GaussianRandomProjection(n_components=n_components, random_state=seed)
        proj_matrix = np.float32(projector._make_random_matrix(n_components,n_features))
        pinv = np.linalg.pinv(proj_matrix)
        projections = tf.matmul(a=flat_images, b=proj_matrix, transpose_b=True)
        inverse_projections = tf.matmul(a=projections, b=pinv, transpose_b=True)
        inverse_projections = tf.reshape(inverse_projections, shape=tf.TensorShape([self.batch_size, rows,cols,channels]))
        proj_logits = self._get_logits(inputs=inverse_projections)
        loss = K.categorical_crossentropy(target=y_true, output=proj_logits)
        loss_gradient = K.gradients(loss=loss, variables=inputs)
        regularization += tf.reduce_sum(tf.square(tf.norm(loss_gradient, ord=2, axis=0)))

        return regularization / self.batch_size #(n_proj*self.batch_size)

    def train(self, x_train, y_train):
        print("\nTraining infos:\nbatch_size = ", self.batch_size, "\nepochs = ", self.epochs,
              "\nx_train.shape = ", x_train.shape, "\ny_train.shape = ", y_train.shape, "\n")

        start_time = time.time()

        self.batches = int(len(x_train)/self.batch_size)
        x_train = np.split(x_train, self.batches)
        y_train = np.split(y_train, self.batches)
        for batch in range(self.batches):
            print("\n=== training batch", batch+1,"/",self.batches,"===")
            inputs = tf.convert_to_tensor(x_train[batch])
            for proj in range(3):
                print("\nprojection",proj+1,"/ 3")
                self.model.compile(loss=self.loss_wrapper(inputs, batch), optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
                self.model.fit(x_train[batch], y_train[batch], epochs=self.epochs, batch_size=self.batch_size) #callbacks=[EpochIdxCallback(self.model)]

        print("\nTraining time: --- %s seconds ---" % (time.time() - start_time))
        return self

    def evaluate_test(self, x_test, y_test):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        :return: x_test predictions
        """

        # todo: set classes_ attribute for loaded models
        if len(y_test.shape) == 2 and y_test.shape[1] > 1:
            self.classes_ = np.arange(y_test.shape[1])
        elif (len(y_test.shape) == 2 and y_test.shape[1] == 1) or len(y_test.shape) == 1:
            self.classes_ = np.unique(y_test)

        print("\n===== Test set evaluation =====")
        print("\nTesting infos:\nx_test.shape = ", x_test.shape, "\ny_test.shape = ", y_test.shape, "\n")

        y_test_pred = self.predict(x_test) # self.predict
        y_test_true = np.argmax(y_test, axis=1)
        correct_preds = np.sum(y_test_pred == np.argmax(y_test, axis=1))

        print("Correctly classified: {}".format(correct_preds))
        print("Incorrectly classified: {}".format(len(x_test) - correct_preds))

        acc = np.sum(y_test_pred == y_test_true) / y_test.shape[0]
        print("Test accuracy: %.2f%%" % (acc * 100))

        # classification report over single classes
        print(classification_report(np.argmax(y_test, axis=1), y_test_pred, labels=range(self.num_classes)))

        return y_test_pred

    def evaluate_adversaries(self, x_test, y_test, method, dataset_name, adversaries_path=None, test=False):
        print("\n===== Adversarial evaluation =====")

        # generate adversaries on the test set
        x_test_adv = self._get_adversaries(self, x_test, y_test, method=method, dataset_name=dataset_name,
                                                adversaries_path=adversaries_path, test=test)
        # debug
        print(len(x_test), len(x_test_adv))

        # evaluate the performance on the adversarial test set
        y_test_adv = self.predict(x_test_adv)
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

    def _get_adversaries(self, trained_classifier, x, y, test, method, dataset_name, adversaries_path):
        """
        Generates adversaries on the input data x using a given method or loads saved data if available.

        :param classifier: trained classifier
        :param x: input data
        :param method: art.attack method
        :param adversaries_path: path of saved pickle data
        :return: adversarially perturbed data
        """
        if adversaries_path is None:
            # todo: buggy...
            classifier = artKerasClassifier((MIN, MAX), trained_classifier.model, use_logits=False)
            print("\nGenerating adversaries with", method, "method on", dataset_name)
            x_adv = None
            if method == 'fgsm':
                attacker = FastGradientMethod(classifier, eps=0.5)
                x_adv = attacker.generate(x=x)
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

class EpochIdxCallback(keras.callbacks.Callback):
    def __init__(self, model):
        super(EpochIdxCallback,self).__init__()
        self.model = model
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        return epoch
    def set_model(self, model):
        return self.model


def main(dataset_name, test, lam):
    # load dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name, test=test)

    # Tensorflow session and initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    classifier = RandomRegularizer(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                   dataset_name=dataset_name, sess=sess, lam=lam, test=test)

    rel_path = time.strftime('%Y-%m-%d') + "/" + str(dataset_name) + "_randreg_weights_lam=" + str(classifier.lam) + ".h5"

    classifier.train(x_train, y_train)
    classifier.model.save_weights(RESULTS + rel_path)
    #classifier.model.load_weights(RESULTS+rel_path)

    # evaluations #
    classifier.evaluate_test(x_test=x_test, y_test=y_test)

    for attack in ['fgsm','pgd','deepfool','carlini_linf']:
        filename = dataset_name + "_x_test_" + attack + ".pkl" #"_randreg.pkl"
        x_test_adv = classifier.evaluate_adversaries(x_test, y_test, method=attack, dataset_name=dataset_name,
                                                     adversaries_path=DATA_PATH+filename,
                                                     test=test)

if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        lam = float(sys.argv[3])

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        lam = input("\nChoose lambda regularization weight.")

    main(dataset_name=dataset_name, test=test, lam=lam)
