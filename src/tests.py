import unittest
from utils import *
from baseline_convnet import BaselineConvnet
from random_ensemble import RandomEnsemble
import time
from projection_functions import *
import random
from random_regularizer import RandomRegularizer
from parallel_randens_training import ParallelRandomEnsemble

BATCH_SIZE = 20
EPOCHS = 1
N_PROJECTIONS = 1
SIZE_PROJECTION = 6
EPS = 0.3
TRAINED_MODELS = "../trained_models/"


class Test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        # self.eps=0.3
        self.dataset = "mnist"
        self.x_train, self.y_train, self.x_test, self.y_test, \
        self.input_shape, self.num_classes, self.data_format = load_dataset(dataset_name="mnist", test=True)
        # baseline on mnist
        self.baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                        data_format=self.data_format, dataset_name=self.dataset, test=True)

    def test_baseline(self):
        model = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                data_format=self.data_format, dataset_name=self.dataset, test=True)

        # model training
        classifier = model.train(self.x_train, self.y_train, batch_size=model.batch_size, epochs=model.epochs)
        model.evaluate(classifier, self.x_test, self.y_test)
        x_test_adv = model.generate_adversaries(classifier, self.x_test, self.y_test, dataset_name=self.dataset,
                                                method="fgsm", eps=EPS, test=True)
        save_to_pickle(data=x_test_adv, filename=self.dataset+"_x_test_fgsm.pkl")
        model.evaluate(classifier, x_test_adv, self.y_test)

        # save and load classifier
        model.save_classifier(classifier=classifier)
        loaded_classifier = model.load_classifier(relative_path=RESULTS+time.strftime('%Y-%m-%d')+"/")
        # loaded_classifier = super(BaselineConvnet, model).load_classifier(
        #     path=RESULTS+time.strftime('%Y-%m-%d') + "/baseline_convnet.h5")

        model.evaluate(loaded_classifier, self.x_test, self.y_test)
        x_test_adv = model.load_adversaries(dataset_name=self.dataset,attack="fgsm",eps=EPS,test=True)
        model.evaluate(loaded_classifier, x_test_adv, self.y_test)

        # complete model loading
        # classifier = model.load_classifier(relative_path=TRAINED_MODELS+"baseline/mnist_baseline.h5")
        classifier = model.load_classifier(relative_path=TRAINED_MODELS+"baseline/")

        # adversarial training
        model.adversarial_train(classifier, dataset_name=model.dataset_name, x_train=self.x_train, y_train=self.y_train,
                                batch_size=BATCH_SIZE, epochs=EPOCHS, method='fgsm', eps=EPS, test=True)

    def test_random_ensemble(self):
        for projection_mode in ["flat","channels","grayscale"]:
            model = RandomEnsemble(input_shape=self.input_shape, num_classes=self.num_classes, dataset_name=self.dataset,
                                   n_proj=N_PROJECTIONS, size_proj=SIZE_PROJECTION, data_format=self.data_format,
                                   projection_mode=projection_mode, test=True)

            # train
            classifiers = model.train(self.x_train, self.y_train, batch_size=model.batch_size, epochs=model.epochs)

        # evaluate
        x_test_pred = model.evaluate(classifiers, self.x_test, self.y_test)
        x_test_adv = self.baseline.load_adversaries(dataset_name=self.dataset,attack="fgsm",eps=EPS,test=True)
        model.evaluate(classifiers, x_test_adv, self.y_test)

        # save and load
        model.save_classifier(classifier=classifiers, model_name="random_ensemble")
        relpath = RESULTS+time.strftime('%Y-%m-%d')+"/"
        loaded_classifiers = model.load_classifier(relative_path=relpath)
        x_test_pred_loaded = model.evaluate(loaded_classifiers, self.x_test, self.y_test)

        # check equal test predictions
        np.array_equal(x_test_pred, x_test_pred_loaded)

    def test_cifar_load_and_train(self):
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_cifar(test=True)
        model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                dataset_name="cifar", test=True)
        model.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    def test_compute_plot_projections(self):
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name="cifar",
                                                                                               test=True)
        for projection_mode in ["flat","channels","grayscale"]:
            random_seeds = random.sample(range(1, 100), N_PROJECTIONS)
            projections, inverse_projections = compute_projections(input_data=x_test, n_proj=N_PROJECTIONS,
                                                                   size_proj=SIZE_PROJECTION,
                                                                   random_seeds=random_seeds,
                                                                   projection_mode=projection_mode)

        plot_projections(image_data_list=[x_test, projections[0], inverse_projections[0]], cmap="gray", test=True)

    def test_cifar_randreg(self):
        dataset_name = "cifar"
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                               test=True)
        for projection_mode in ["no_projections","loss_on_projections","projected_loss"]:
            classifier = RandomRegularizer(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                           dataset_name=dataset_name, sess=sess, lam=0.6,
                                           projection_mode=projection_mode, test=True)
            classifier.train(x_train, y_train)
            classifier.evaluate(x=x_test, y=y_test)

    def test_parallel_randens(self):
        dataset_name=self.dataset
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test=True)

        model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=SIZE_PROJECTION,
                                       data_format=data_format, dataset_name=dataset_name, projection_mode="flat")
        model.train_single_projection(x_train=x_train, y_train=y_train, batch_size=model.batch_size,
                                      epochs=model.epochs, idx=1, save=False)


if __name__ == '__main__':
    unittest.main()
