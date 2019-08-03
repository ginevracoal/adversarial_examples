# -*- coding: utf-8 -*-

"""
Simple CNN model. This is our benchmark on the MNIST dataset.
"""

from adversarial_classifier import AdversarialClassifier
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from utils import *
import time
import sys
import matplotlib.pyplot as plt

###############
# main() args #
###############
DATASET="cifar"
TEST=True
ATTACK="fgsm"


####################
# default settings #
####################
MODEL_NAME = "baseline_convnet"
TRAINED_MODELS = "../trained_models/"
DATA_PATH = "../data/"
RESULTS = "../results/"+time.strftime('%Y-%m-%d')+"/"
BATCH_SIZE = 128
EPOCHS = 12


class BaselineConvnet(AdversarialClassifier):

    def _set_layers(self):
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


def main(dataset, test, attack):
    """
    :param dataset: choose between "mnist" and "cifar"
    :param test: if True, only takes the first 100 samples.
    :param attack: choose between "fgsm", "pgd", "deepfool", "carlini_linf", "virtual", "newtonfool"
    """

    # load dataset #
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset, test=test)
    #plt.imshow(x_test[5])

    # train classifier #
    model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format)
    classifier = model.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    model.save_model(classifier = classifier, model_name = dataset+"_baseline")

    # load classifier #
    #classifier = model.load_classifier(relative_path=TRAINED_MODELS+"baseline/"+DATASET+"_baseline.h5")
    #classifier = model.load_classifier(relative_path=TRAINED_MODELS+"baseline/"+DATASET+"_"+ATTACK+"_robust_baseline.h5")

    # adversarial training #
    #robust_classifier = model.adversarial_train(classifier, x_train, y_train, x_test, y_test, test=TEST,
    #                                                         batch_size=BATCH_SIZE, epochs=EPOCHS, method=ATTACK)
    #model.save_model(classifier = robust_classifier, model_name = DATASET+"_"+ATTACK+"_robust_baseline")

    # evaluations #
    model.evaluate_test(classifier, x_test, y_test)

    #############
    # todo: solve this bug eventually... not urgent
    # notice: here we are actually not only saving x_test, but also y_test... This is not a problem since pkl
    # loading deals with this issue. the correct code should be:
    # x_test_adv, _ = model.evaluate_adversaries(...)
    #############

    #x_test_adv = model.evaluate_adversaries(classifier, x_test, y_test, method=attack, test=test,
                                            # adversaries_path=DATA_PATH+dataset+"_x_test_"+attack+".pkl"
    #                                        )
    #save_to_pickle(data=x_test_adv, filename=dataset+"_x_test_"+attack+".pkl")

    #plt.imshow(x_test_adv[5])
    #plt.show()


if __name__ == "__main__":
    # try:
    #     dataset = sys.argv[1]
    #     test = sys.argv[2]
    #     attack = sys.argv[3]
    #
    # except IndexError:
    #     dataset = input("\nChoose a dataset.")
    #     test = input("\nDo you just want to test the code?")
    #     attack = input("\nChoose an attack.")

    main(DATASET, TEST, ATTACK)
