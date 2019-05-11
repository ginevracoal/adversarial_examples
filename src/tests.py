import unittest
from baseline_convnet import BaselineConvnet
from random_ensemble import RandomEnsemble
from utils import *


BATCH_SIZE = 20
EPOCHS = 5
N_PROJECTIONS = 3
SIZE_PROJECTION = 8


class Test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.x_train, self.y_train, self.x_test, self.y_test, \
            self.input_shape, self.num_classes = self._load_small_mnist(size=100)

    def _load_small_mnist(self, size):
        x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_mnist()
        return x_train[:size], y_train[:size], x_test[:size], y_test[:size], input_shape, num_classes

    def test_baseline(self):
        convNet = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)

        # model training
        classifier = convNet.train(self.x_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
        convNet.evaluate_test(classifier, self.x_test, self.y_test)
        convNet.evaluate_adversaries(classifier, self.x_test, self.y_test)

        # model loading
        classifier = convNet.load_classifier("IBM-art/mnist_cnn_original.h5")
        convNet.evaluate_test(classifier, self.x_test, self.y_test)
        convNet.evaluate_adversaries(classifier, self.x_test, self.y_test)

    def test_random_ensemble(self):
        model = RandomEnsemble(input_shape=self.input_shape, num_classes=self.num_classes,
                               n_proj=N_PROJECTIONS, size_proj=SIZE_PROJECTION)

        classifier = model.train(self.x_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
        model.evaluate_test(classifier, self.x_test, self.y_test)
        model.evaluate_adversaries(classifier, self.x_test, self.y_test)

    def test_projector_shapes(self):
        # TODO: aggiungere test sulle dimensioni calcolate da questa funzione
        # print(predictions.shape, summed.shape, argmax_predictions.shape) # (3,100,10), (100,10), 100

        pass

    def test_no_projector_available(self):
        pass


if __name__ == '__main__':
    unittest.main()
