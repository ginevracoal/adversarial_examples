import unittest
from baseline_convnet import BaselineConvnet
from utils import *


MODEL_NAME = "baseline_convnet"
TRAINED_MODEL = "IBM-art/mnist_cnn_original.h5"

BATCH_SIZE = 20
EPOCHS = 5


class Test(unittest.TestCase):

    def test_baseline_training(self):
        x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_mnist()

        # take a subset
        print("\nTaking just the first 100 images.")
        x_train, y_train, x_test, y_test = x_train[:100], y_train[:100], x_test[:100], y_test[:100]

        convNet = BaselineConvnet(input_shape=input_shape, num_classes=num_classes)

        # model training
        classifier = convNet.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
        convNet.evaluate_test(classifier, x_test, y_test)
        convNet.evaluate_adversaries(classifier, x_test, y_test)

        # model loading
        classifier = convNet.load_classifier(TRAINED_MODEL)
        convNet.evaluate_test(classifier, x_test, y_test)
        convNet.evaluate_adversaries(classifier, x_test, y_test)


if __name__ == '__main__':
    unittest.main()
