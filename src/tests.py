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
        self.dataset = "mnist"
        self.x_train, self.y_train, self.x_test, self.y_test, \
            self.input_shape, self.num_classes, self.data_format = load_dataset(dataset_name="mnist", test=True)
        # baseline on mnist
        self.baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                        data_format=self.data_format, dataset_name=self.dataset)

    def test_baseline(self):
        model = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                data_format=self.data_format, dataset_name=self.dataset)

        # model training
        classifier = model.train(self.x_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
        model.evaluate_test(classifier, self.x_test, self.y_test)
        x_test_adv = model.evaluate_adversaries(classifier, self.x_test, self.y_test, dataset_name=self.dataset,
                                                method="fgsm")

        # save to pickle
        save_to_pickle(data=x_test_adv, filename=self.dataset+"_x_test_fgsm.pkl")

        # save and load
        model.save_model(classifier=classifier, model_name="baseline_convnet")
        loaded_classifier = model.load_classifier(
            relative_path=RESULTS+time.strftime('%Y-%m-%d') + "/baseline_convnet.h5")

        model.evaluate_test(loaded_classifier, self.x_test, self.y_test)
        model.evaluate_adversaries(loaded_classifier, self.x_test, self.y_test, test=True, dataset_name=self.dataset,
                                   adversaries_path=RESULTS+time.strftime('%Y-%m-%d')+"/mnist_x_test_fgsm.pkl",
                                   method="fgsm")

        # complete model loading
        classifier = model.load_classifier(relative_path=TRAINED_MODELS+"baseline/mnist_baseline.h5")

        model.evaluate_adversaries(classifier, self.x_test, self.y_test, dataset_name=self.dataset,
                                   method='deepfool', adversaries_path='../data/mnist_x_test_deepfool.pkl', test=True)

        # adversarial training
        robust_classifier = model.adversarial_train(classifier, self.x_train, self.y_train, self.x_test, self.y_test,
                                                    batch_size=BATCH_SIZE, epochs=EPOCHS, method='fgsm', test=True,
                                                    dataset_name=self.dataset)
        model.evaluate_adversaries(robust_classifier, self.x_test, self.y_test, test=True, dataset_name=self.dataset,
                                   adversaries_path=RESULTS + time.strftime('%Y-%m-%d') + "/mnist_x_test_fgsm.pkl",
                                   method="fgsm")

    def test_random_ensemble(self):
        model = RandomEnsemble(input_shape=self.input_shape, num_classes=self.num_classes, dataset_name=self.dataset,
                               n_proj=N_PROJECTIONS, size_proj=SIZE_PROJECTION, data_format=self.data_format)

        # train
        classifiers = model.train(self.x_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
        model.save_model(classifier=classifiers, model_name="random_ensemble")

        # evaluate
        x_test_pred = model.evaluate_test(classifiers, self.x_test, self.y_test)
        model.evaluate_adversaries(classifiers, self.x_test, self.y_test, method='fgsm', dataset_name=self.dataset)

        # save and load
        model.save_model(classifier=classifiers, model_name="random_ensemble"#proj=" + str(model.n_proj) +
                                                            "_size=" + str(model.size_proj))
        loaded_classifiers = model.load_classifier(relative_path=RESULTS+time.strftime('%Y-%m-%d')+"/",
                                                   model_name="random_ensemble")
        x_test_pred_loaded = model.evaluate_test(loaded_classifiers, self.x_test, self.y_test)

        # check equal test predictions
        np.array_equal(x_test_pred, x_test_pred_loaded)

    def test_cifar_load_and_train(self):
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_cifar(test=True)
        model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                dataset_name=self.dataset)
        model.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

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
