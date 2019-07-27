# -*- coding: utf-8 -*-

"""
Simple CNN model. This is our benchmark on the MNIST dataset.
"""

from adversarial_classifier import AdversarialClassifier
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from utils import *
import time

SAVE = False
TEST = True

MODEL_NAME = "baseline_convnet"
TRAINED_MODELS = "../trained_models/"
TRAINED_MODEL = TRAINED_MODELS+"baseline/baseline.h5"
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
        # model.summary()
        return model

def main():

    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_mnist(test=TEST)

    model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes)
    classifier = model.load_classifier(relative_path=TRAINED_MODELS+"baseline/baseline.h5")

    # classifier = model.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    model.evaluate_adversaries(classifier, x_test, y_test, method='fgsm', test=TEST,
                                             adversaries_path="../data/mnist_x_test_fgsm.pkl")

    robust_classifier = model.adversarial_train(classifier, x_train, y_train, x_test, y_test, test=TEST,
                                                batch_size=BATCH_SIZE, epochs=EPOCHS, method='fgsm')
    model.save_model(classifier=robust_classifier, model_name="baseline_robust")

    # todo: debug, this output should be the same as the one given at the end of adversarial training! check
    print("\nEval on x_test_fgsm.pkl")
    model.evaluate_adversaries(robust_classifier, x_test, y_test, method='fgsm', test=TEST,
                                             adversaries_path="../data/mnist_x_test_fgsm.pkl")

    #save_to_pickle(data=x_test_fgsm, filename="mnist_x_test_fgsm.pkl")

    #x_test_deepfool = model.evaluate_adversaries(classifier, x_test, y_test, method='deepfool', test=TEST,
    #                                             adversaries_path='../data/mnist_x_test_deepfool.pkl')
    #save_to_pickle(data=x_test_deepfool, filename="mnist_x_test_deepfool.pkl")

    #x_test_pgd = model.evaluate_adversaries(classifier, x_test, y_test, method='projected_gradient', test=TEST,
    #                                        adversaries_path='../data/mnist_x_test_projected_gradient.pkl')
    #save_to_pickle(data=x_test_pgd, filename="mnist_x_test_pgd.pkl")

    #x_test_carlini = model.evaluate_adversaries(classifier, x_test, y_test, method='carlini_linf', test=TEST,
    #                                            adversaries_path=DATA_PATH+'mnist_x_test_carlini.pkl')
    #save_to_pickle(data=x_test_carlini, filename="mnist_x_test_carlini.pkl")

    # x_test_virtual = model.evaluate_adversaries(classifier, x_test, y_test, method='virtual_adversarial')
    # x_test_newtonfool = model.evaluate_adversaries(classifier, x_test, y_test, method='newtonfool')


if __name__ == "__main__":
    main()
