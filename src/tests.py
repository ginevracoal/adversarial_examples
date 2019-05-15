import unittest
from utils import *
from baseline_convnet import BaselineConvnet
from random_ensemble import RandomEnsemble
import time

BATCH_SIZE = 20
EPOCHS = 3
N_PROJECTIONS = 2
SIZE_PROJECTION = 6

TRAINED_MODELS = "../trained_models/"


class Test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.x_train, self.y_train, self.x_test, self.y_test, \
            self.input_shape, self.num_classes = preprocess_mnist(test=True)
        self.baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)

    def test_baseline_training(self):
        convNet = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)

        # model training
        classifier = convNet.train(self.x_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
        convNet.evaluate_test(classifier, self.x_test, self.y_test)
        convNet.evaluate_adversaries(classifier, self.x_test, self.y_test)

        # model loading
        classifier = convNet.load_classifier(relative_path="IBM-art/mnist_cnn_original.h5")

        # adversarial training
        convNet.adversarial_train(classifier, self.x_train, self.y_train, self.x_test, self.y_test,
                                  batch_size=BATCH_SIZE, epochs=EPOCHS, method='fgsm')

    def test_random_ensemble(self):
        model = RandomEnsemble(input_shape=self.input_shape, num_classes=self.num_classes,
                               n_proj=N_PROJECTIONS, size_proj=SIZE_PROJECTION)

        # train
        classifiers = model.train(self.x_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

        # evaluate
        model.evaluate_test(classifiers, self.x_test, self.y_test)
        model.evaluate_adversaries(classifiers, self.x_test, self.y_test)
        model.evaluate_adversaries(classifiers, self.x_test, self.y_test, method='deepfool')

        # save and load
        model.save_model(classifier=classifiers, model_name="random_ensemble")
        model.load_classifier(relative_path=TRAINED_MODELS+time.strftime('%Y-%m-%d')+"/")

    def test_random_adversarial_projection(self):
        pass
        """
        model = RandomAdversarialProjection(input_shape=self.input_shape, num_classes=self.num_classes,
                                            n_proj=1, size_proj=SIZE_PROJECTION)

        classifier = model.train(self.x_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
        model.evaluate_test(classifier, self.x_test, self.y_test)
        model.evaluate_adversaries(classifier, self.x_test, self.y_test)

        # classifier = model.load_classifier(TRAINED_MODEL)
        model.evaluate_test(classifier, self.x_test, self.y_test)
        model.evaluate_adversaries(classifier, self.x_test, self.y_test)
        """


if __name__ == '__main__':
    unittest.main()
