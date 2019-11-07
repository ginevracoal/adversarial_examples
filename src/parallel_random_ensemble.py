# -*- coding: utf-8 -*-

from random_ensemble import *
from projection_functions import compute_single_projection
from utils import load_dataset
from joblib import Parallel, delayed
import multiprocessing

MODEL_NAME = "random_ensemble"
PROJ_MODE = "channels"
DATASETS = "mnist, cifar"


class ParallelRandomEnsemble(RandomEnsemble):

    def __init__(self, input_shape, num_classes, size_proj, proj_idx, n_proj, data_format, dataset_name,
                 projection_mode, test, epochs=None):
        super(ParallelRandomEnsemble, self).__init__(input_shape=input_shape, num_classes=num_classes, n_proj=n_proj,
                                                     size_proj=size_proj, projection_mode=projection_mode,
                                                     data_format=data_format, dataset_name=dataset_name, test=test,
                                                     epochs=epochs)
        self.proj_idx = proj_idx
        self.n_jobs = 2  # multiprocessing.cpu_count()

    @staticmethod
    def _set_session(device):
        return None

    def train_single_projection(self, x_train, y_train, device, proj_idx):
        """ Trains a single projection of the ensemble classifier and saves the model in current day results folder."""

        print("\nTraining single randens projection with seed =", str(proj_idx),
              "and size_proj =", str(self.size_proj))

        start_time = time.time()
        x_train_projected, x_train_inverse_projected = compute_single_projection(input_data=x_train,
                                                                                 seed=proj_idx,
                                                                                 size_proj=self.size_proj,
                                                                                 projection_mode=self.projection_mode)
        # eventually adjust input dimension to a single channel projection
        if x_train_projected.shape[3] == 1:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 1)

        # use the same model architecture (not weights) for all trainings
        proj_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                   data_format=self.data_format, dataset_name=self.dataset_name, test=self.test)
        proj_classifier.train(x_train_projected, y_train, device)
        print("\nProjection + training time: --- %s seconds ---" % (time.time() - start_time))
        self.trained = True
        # todo: bug in saving when using parallel implementation
        proj_classifier.save_classifier(relative_path=RESULTS, folder=self.folder,
                                        filename=self._set_baseline_filename(seed=self.proj_idx))
        return proj_classifier

    def train(self, x, y, device):
        self.classifiers = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_train)(dataset_name=self.dataset_name, test=self.test, proj_idx=proj_idx,
                                     size_proj=self.size_proj, proj_mode=self.projection_mode, device=device)
            for proj_idx in range(self.n_proj))
        self.trained = True

    def compute_projections(self, input_data):
        """
        Parallel implementation of this method
        :param input_data:
        :return:
        """
        projections = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_projections)(input_data, proj_idx=proj_idx, size_proj=self.size_proj,
                                                   projection_mode=self.projection_mode)
            for proj_idx in self.random_seeds)

        # eventually adjust input dimension to a single channel projection
        projections = np.array(projections)

        if projections.shape[4] == 1:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 1)
        else:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 3)
        return projections, None

    def _sum_ensemble_classifier(self, classifiers, projected_data):
        """
        Parallelized version of this method.
        :param classifiers: list of trained classifiers
        :param projected_data: list of projected data for all of the n_prok random initializations
        :return:
        """
        # compute predictions for each projection
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_predict)(classifier=classifier, projected_data=projected_data[i], device=device)
            for i, classifier in enumerate(classifiers))
        proj_predictions = np.array(results)
        # sum the probabilities across all predictors
        predictions = np.sum(proj_predictions, axis=0)
        return predictions

    # def evaluate(self, x, y, report_projections=False):
    #     """
    #     Performs parallel evaluation of the ensemble model on cpu.
    #     :param x: evaluation data
    #     :param y: evaluation labels
    #     :return: model predictions
    #     """
    #     super(ParallelRandomEnsemble, self).evaluate(x, y)

    # def load_classifier(self, relative_path, folder=None, filename=None):
    #     classifiers = super(ParallelRandomEnsemble, self).load_classifier(relative_path, folder, filename)
    #     for classifier in classifiers:
    #         classifier.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
    #                                metrics=['accuracy'])
    #         # classifier.model._make_predict_function()
    #     return classifiers

def _set_session(device):
    """ Initialize tf session """
    # print(device_lib.list_local_devices())
    from keras.backend.tensorflow_backend import set_session
    if device == "gpu":
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.allow_soft_placement = True
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras
        sess.run(tf.global_variables_initializer())
        # print("check cuda: ", tf.test.is_built_with_cuda())
        # print("check gpu: ", tf.test.is_gpu_available())
        return sess


def _parallel_predict(classifier, projected_data, device="cpu"):
    import tensorflow as tf
    _set_session(device)
    # use the same computational graph of training for the predictions
    g = tf.get_default_graph()
    with g.as_default():
        predictions = classifier.predict(projected_data)
    del tf
    return predictions


def _parallel_train(dataset_name, test, proj_idx, size_proj, proj_mode, device):
    import tensorflow as tf
    _set_session(device)
    g = tf.Graph()
    with g.as_default():
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)

        model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj,
                                       proj_idx=proj_idx, data_format=data_format, dataset_name=dataset_name,
                                       projection_mode=proj_mode, test=test, n_proj=1)
        model.train_single_projection(x_train=x_train, y_train=y_train, device=device, proj_idx=proj_idx)
    del tf


def  _parallel_compute_projections(input_data, proj_idx, size_proj, projection_mode):
    projection, _ = compute_single_projection(input_data=input_data, seed=proj_idx,
                                                               size_proj=size_proj, projection_mode=projection_mode)
    return projection


def main(dataset_name, test, proj_idx, n_proj, size_proj, proj_mode, device):

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)

    # === train and save a single projection === #
    # model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj,
    #                                proj_idx=proj_idx, n_proj=1, data_format=data_format, dataset_name=dataset_name,
    #                                projection_mode=proj_mode, test=test)
    # model.train_single_projection(x_train, y_train, proj_idx=proj_idx, device=device)

    # === parallel train and save the whole ensemble === #
    model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj,
                                   proj_idx=None, n_proj=n_proj, data_format=data_format, dataset_name=dataset_name,
                                   projection_mode=proj_mode, test=test)
    # model.train(x_train, y_train, device=device)
    model.load_classifier(relative_path=RESULTS)
    # model.load_classifier(relative_path=TRAINED_MODELS)

    # === parallel evaluate === #
    model.evaluate(x=x_test, y=y_test)
    for method in ['fgsm', 'pgd', 'carlini']:
        x_test_adv = model.load_adversaries(attack=method, eps=0.3)
        model.evaluate(x_test_adv, y_test)
    for method in ['fgsm', 'pgd', 'carlini']:
        x_test_adv = model.load_adversaries(attack=method, eps=0.5)
        model.evaluate(x_test_adv, y_test)

    x_test_adv = model.load_adversaries(attack="deepfool", eps=None)
    model.evaluate(x_test_adv, y_test)


if __name__ == "__main__":

    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        proj_idx = int(sys.argv[3])
        n_proj = int(sys.argv[4])
        size_proj_list = list(map(int, sys.argv[5].strip('[]').split(',')))
        projection_mode = sys.argv[6]
        device = sys.argv[7]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        proj_idx = input("\nSet a seed for projections (int): ")
        n_proj = input("\nChoose the number of projections (type=list): ")
        size_proj_list = input("\nChoose size for the projection (list): ")
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")
        device = input("\nChoose a device (cpu/gpu): ")

    for size_proj in size_proj_list:
        main(dataset_name=dataset_name, test=test, proj_idx=proj_idx, n_proj=n_proj, size_proj=size_proj,
             proj_mode=projection_mode, device=device)

