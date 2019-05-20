# -*- coding: utf-8 -*-

"""
Simple CNN model. This is out benchmark on the MNIST dataset.
"""

from adversarial_classifier import AdversarialClassifier
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from utils import *

SAVE = False
TEST = True

MODEL_NAME = "baseline_convnet"
TRAINED_MODEL = "IBM-art/mnist_cnn_original.h5"
DATA_PATH = "../data/"
RESULTS = "../results/"+time.strftime('%Y-%m-%d')+"/"

BATCH_SIZE = 128
EPOCHS = 12


class BaselineConvnet(AdversarialClassifier):

    def _set_layers(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, kernel_size=(3, 3),
                   activation='relu', data_format='channels_last')(inputs)
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
        #model.summary()
        return model


def main():

    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_mnist(test=TEST)

    model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes)
    classifier = model.load_classifier(relative_path=TRAINED_MODEL)

    x_test_virtual = model.evaluate_adversaries(classifier, x_test, y_test, method='virtual_adversarial')
    x_test_carlini = model.evaluate_adversaries(classifier, x_test, y_test, method='carlini_l2')
    x_test_projected_gradient = model.evaluate_adversaries(classifier, x_test, y_test, method='projected_gradient')
    x_test_newtonfool = model.evaluate_adversaries(classifier, x_test, y_test, method='newtonfool')

    if SAVE is True:

        save_to_pickle(data=x_test_virtual, filename="mnist_x_test_virtual.pkl")
        save_to_pickle(data=x_test_carlini, filename="mnist_x_test_carlini.pkl")
        save_to_pickle(data=x_test_projected_gradient, filename="mnist_x_test_projected_gradient.pkl")
        save_to_pickle(data=x_test_newtonfool, filename="mnist_x_test_newtonfool.pkl")


if __name__ == "__main__":
    main()
