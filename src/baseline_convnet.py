# -*- coding: utf-8 -*-

"""
Simple CNN model. This is our benchmark on the MNIST dataset.
"""

from adversarial_classifier import AdversarialClassifier
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from utils import *
import time
import sys
import matplotlib.pyplot as plt

####################
# default settings #
####################
MODEL_NAME = "baseline_convnet"
TRAINED_MODELS = "../trained_models/"
DATA_PATH = "../data/"
RESULTS = "../results/"+time.strftime('%Y-%m-%d')+"/"


class BaselineConvnet(AdversarialClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        self.dataset_name = dataset_name
        self.batch_size, self.epochs = self._set_training_params()
        super(BaselineConvnet, self).__init__(input_shape, num_classes, data_format)

    def _set_training_params(self):
        if self.dataset_name == "mnist":
            return 128, 12
        elif self.dataset_name == "cifar":
            return 128, 120

    def _set_layers(self):

        if self.dataset_name == "mnist":

            inputs = Input(shape=self.input_shape)
            x = Conv2D(32, kernel_size=(3, 3),
                       activation='relu', data_format=self.data_format)(inputs)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(self.num_classes, activation='softmax')(x)

            model = Model(inputs=inputs, outputs=predictions)
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
            # model.summary()
            return model

        elif self.dataset_name == "cifar":

            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                             input_shape=self.input_shape))
                             #input_shape=(32, 32, 3)))
            model.add(BatchNormalization())
            model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(BatchNormalization())
            model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.3))
            model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(BatchNormalization())
            model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.4))
            model.add(Flatten())
            model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='softmax'))
            # compile model
            opt = SGD(lr=0.001, momentum=0.9)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            return model


def main(dataset_name, test, attack):
    """
    :param dataset: choose between "mnist" and "cifar"
    :param test: if True, only takes the first 100 samples.
    :param attack: choose between "fgsm", "pgd", "deepfool", "carlini_linf", "virtual", "newtonfool"
    """

    # load dataset #
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name, test=test)
    #plt.imshow(x_test[5])

    # train classifier #
    model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format, dataset_name=dataset_name)
    #classifier = model.train(x_train, y_train, batch_size=model.batch_size, epochs=model.epochs)
    #model.save_model(classifier = classifier, model_name = dataset_name+"_baseline")

    # load classifier #
    classifier = model.load_classifier(relative_path=TRAINED_MODELS+"baseline/"+dataset_name+"_baseline.h5")
    #classifier = model.load_classifier(relative_path=TRAINED_MODELS+"baseline/"+dataset_name+"_"+attack+"_robust_baseline.h5")

    # adversarial training #
    robust_classifier = model.adversarial_train(classifier, x_train, y_train, test=test, method=attack,
                                                batch_size=model.batch_size, epochs=model.epochs, dataset_name=dataset_name)
    model.save_model(classifier = robust_classifier, model_name = dataset_name+"_"+attack+"_robust_baseline")

    # evaluations #
    model.evaluate_test(classifier, x_test, y_test)

    #############
    # todo: solve this bug eventually... not urgent
    # notice: here we are actually not only saving x_test, but also y_test... This is not a problem since pkl
    # loading deals with this issue. the correct code should be:
    # x_test_adv, _ = model.evaluate_adversaries(...)
    #############

    #x_test_adv = model.evaluate_adversaries(classifier=classifier, x_test=x_test, y_test=y_test,
    #                                        method=attack, test=test, dataset_name=dataset_name)
    #save_to_pickle(data=x_test_adv, filename=dataset_name+"_x_test_"+attack+".pkl")

    #for attack in ['fgsm','pgd','deepfool','carlini_linf']:
    #    x_test_adv = model.evaluate_adversaries(classifier, x_test, y_test, method=attack, dataset_name=dataset_name,
    #                                            adversaries_path=DATA_PATH+dataset_name+"_x_test_"+attack+".pkl", test=test)

    #plt.imshow(x_test_adv[5])
    #plt.show()


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        attack = sys.argv[3]

    except IndexError:
        dataset_name = input("\nChoose a dataset.")
        test = input("\nDo you just want to test the code?")
        attack = input("\nChoose an attack.")

    K.clear_session()
    main(dataset_name=dataset_name, test=test, attack=attack)
