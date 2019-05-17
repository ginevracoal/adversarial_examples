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

    def test_baseline(self):
        model = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)

        # model training
        classifier = model.train(self.x_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
        model.evaluate_test(classifier, self.x_test, self.y_test)
        x_test_adv, x_test_adv_pred = model.evaluate_adversaries(classifier, self.x_test, self.y_test)

        # model loading
        classifier = model.load_classifier(relative_path="IBM-art/mnist_cnn_original.h5")

        x_test_adv, x_test_adv_pred = model.evaluate_adversaries(classifier, self.x_test, self.y_test,
                                                                 method='deepfool',
                                                                 adversaries_path='../data/mnist_x_test_deepfool.pkl')
        # save to pickle
        save_to_pickle(data=x_test_adv, filename="mnist_x_test_deepfool.pkl")

        # adversarial training
        model.adversarial_train(classifier, self.x_train, self.y_train, self.x_test, self.y_test,
                                  batch_size=BATCH_SIZE, epochs=EPOCHS, method='fgsm')

    def test_random_ensemble(self):
        model = RandomEnsemble(input_shape=self.input_shape, num_classes=self.num_classes,
                               n_proj=N_PROJECTIONS, size_proj=SIZE_PROJECTION)

        # train
        classifiers = model.train(self.x_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

        # evaluate
        model.evaluate_test(classifiers, self.x_test, self.y_test)
        x_test_adv, x_test_adv_pred = model.evaluate_adversaries(classifiers, self.x_test, self.y_test, method='fgsm')

        # save and load
        model.save_model(classifier=classifiers, model_name="random_ensemble")
        loaded_classifiers = model.load_classifier(relative_path=RESULTS+time.strftime('%Y-%m-%d')+"/")

        # buggy
        # x_test_adv, x_test_adv_pred = model.evaluate_adversaries(loaded_classifiers, self.x_test, self.y_test,
        #                                                         method='deepfool', adversaries_path='../data/mnist_x_test_deepfool.pkl')

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
