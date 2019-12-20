import unittest
import sys
sys.path.append(".")
from directories import *
from utils import *
from RandomProjections.baseline_convnet import BaselineConvnet
from RandomProjections.random_ensemble import RandomEnsemble
from RandomProjections.projection_functions import *
from RandomProjections.random_regularizer import RandomRegularizer
from RandomProjections.parallel_random_ensemble import ParallelRandomEnsemble
from RandomProjections.ensemble_regularizer import EnsembleRegularizer

BATCH_SIZE = 20
EPOCHS = 1
N_PROJECTIONS = 1
SIZE_PROJECTION = 6
DEVICE = "cpu"


class Test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.dataset = "mnist"
        self.x_train, self.y_train, self.x_test, self.y_test, \
        self.input_shape, self.num_classes, self.data_format = load_dataset(dataset_name="mnist", test=True,
                                                                            data=DATA_PATH)
        # baseline on mnist
        self.baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                        data_format=self.data_format, dataset_name=self.dataset, test=True)

    def test_baseline(self):
        model = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                data_format=self.data_format, dataset_name=self.dataset, test=True)

        # model training
        model.train(self.x_train, self.y_train, device=DEVICE)
        model.evaluate(self.x_test, self.y_test)
        x_test_adv = model.generate_adversaries(self.x_test, self.y_test, attack="fgsm", seed=0)
        model.save_adversaries(data=x_test_adv,attack="fgsm", seed=0)
        model.evaluate(x_test_adv, self.y_test)

        # save and load classifier
        model.save_classifier(relative_path=RESULTS)
        model.load_classifier(relative_path=RESULTS)
        model.evaluate(self.x_test, self.y_test)
        x_test_adv = model.load_adversaries(attack="fgsm",seed=0,relative_path=RESULTS)
        model.evaluate(x_test_adv, self.y_test)

        # adversarial training
        model.adversarial_train(x_train=self.x_train, y_train=self.y_train, device="cpu", attack='fgsm')

    def test_random_ensemble(self):
        for projection_mode in ["flat","channels","grayscale"]:
            model = RandomEnsemble(input_shape=self.input_shape, num_classes=self.num_classes, dataset_name=self.dataset,
                                   n_proj=N_PROJECTIONS, size_proj=SIZE_PROJECTION, data_format=self.data_format,
                                   projection_mode=projection_mode, test=True, library="cleverhans")
            # train
            model.train(self.x_train, self.y_train, device="cpu")

        # evaluate
        x_test_pred = model.evaluate(self.x_test, self.y_test)
        x_test_adv = self.baseline.load_adversaries(attack="fgsm",seed=0,relative_path=RESULTS)
        model.evaluate(x_test_adv, self.y_test)

        # save and load
        model.save_classifier(relative_path=RESULTS)
        del model
        model = RandomEnsemble(input_shape=self.input_shape, num_classes=self.num_classes, dataset_name=self.dataset,
                               n_proj=N_PROJECTIONS, size_proj=SIZE_PROJECTION, data_format=self.data_format,
                               projection_mode=projection_mode, test=True, library="art")
        model.load_classifier(relative_path=RESULTS)
        x_test_pred_loaded = model.evaluate(self.x_test, self.y_test)

        # check equal test predictions
        np.array_equal(x_test_pred, x_test_pred_loaded)

    def test_cifar_load_and_train(self):
        print(os.getcwd())
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_cifar(test=True, data=DATA_PATH)
        model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                dataset_name="cifar", test=True)
        model.train(x_train, y_train, device="cpu")

    def test_compute_plot_projections(self):
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name="mnist",
                                                                                               test=True, data=DATA_PATH)
        for projection_mode in ["flat","channels","grayscale"]:
            random_seeds = range(N_PROJECTIONS)  # random.sample(range(1, 100), N_PROJECTIONS)
            projections, inverse_projections = compute_projections(input_data=x_test, n_proj=N_PROJECTIONS,
                                                                   size_proj=SIZE_PROJECTION,
                                                                   random_seeds=random_seeds,
                                                                   projection_mode=projection_mode)
        plot_images(image_data_list=[x_test, projections[0], inverse_projections[0]], cmap="gray", test=True)

    def test_cifar_randreg(self):
        dataset_name = "cifar"
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                               test=True, data=DATA_PATH)
        model = RandomRegularizer(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                  dataset_name=dataset_name, lam=0.6, projection_mode="no_projections", test=True,
                                  library="art")
        model.train(x_train, y_train, device="cpu")
        model.evaluate(x=x_test, y=y_test)

    def test_projection_modes(self):
        dataset_name = "mnist"
        for projection_mode in ["no_projections", "loss_on_projections", "projected_loss"]:
            model = RandomRegularizer(input_shape=self.input_shape, num_classes=self.num_classes,
                                      data_format=self.data_format, dataset_name=dataset_name, lam=0.6,
                                      projection_mode=projection_mode, test=True, library="art")
            model.train(self.x_train, self.y_train, device="cpu")
            model.evaluate(x=self.x_test, y=self.y_test)

    def test_ensemble_regularizer(self):
        ensemble_size=2
        model = EnsembleRegularizer(ensemble_size=ensemble_size, input_shape=self.input_shape,
                                    num_classes=self.num_classes, data_format=self.data_format, dataset_name="mnist",
                                    lam=0.3, projection_mode="loss_on_projections", test=True)
        model.train(self.x_train, self.y_train, device="cpu")
        model.save_classifier(relative_path=RESULTS)
        del model
        # model = EnsembleRegularizer(ensemble_size=ensemble_size, input_shape=self.input_shape,
        #                             num_classes=self.num_classes, data_format=self.data_format, dataset_name="mnist",
        #                             lam=0.3, projection_mode="loss_on_projections", test=True)
        # model.load_classifier(relative_path=RESULTS)
        # model.evaluate(x=self.x_test, y=self.y_test)
        # todo: working on the implementation
        pass

    def test_parallel_randens(self):
        model = ParallelRandomEnsemble(input_shape=self.input_shape, num_classes=self.num_classes,
                                       size_proj=SIZE_PROJECTION, proj_idx=None, n_proj=2,
                                       data_format=self.data_format, dataset_name="mnist",
                                       projection_mode="flat", test=True, library="art")
        model.train(x_train=self.x_train, y_train=self.y_train, device=DEVICE, n_jobs=2)
        model_path = RESULTS
        model.evaluate(x=self.x_test, y=self.y_test, device=DEVICE, model_path=model_path)
        x_test_adv = model.load_adversaries(attack="fgsm", seed=0, relative_path=RESULTS)
        model.evaluate(x=x_test_adv, y=self.y_test, device=DEVICE, model_path=model_path)

    def test_weighted_ensemble(self):
        pass

    def test_robustness_measures(self):
        pass

    def test_centroid_translation(self):
        pass

if __name__ == '__main__':
    unittest.main()
