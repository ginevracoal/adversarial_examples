# -*- coding: utf-8 -*-

from random_ensemble import *
from projection_functions import compute_single_projection
from utils import load_dataset
from joblib import Parallel, delayed

MODEL_NAME = "random_ensemble"
PROJ_MODE = "channels"
DATASETS = "mnist, cifar"


class ParallelRandomEnsemble(RandomEnsemble):

    def __init__(self, input_shape, num_classes, size_proj, proj_idx, data_format, dataset_name, projection_mode, test,
                 epochs=None):
        super(ParallelRandomEnsemble, self).__init__(input_shape=input_shape, num_classes=num_classes, n_proj=1,
                                                     size_proj=size_proj, projection_mode=projection_mode,
                                                     data_format=data_format, dataset_name=dataset_name, test=test,
                                                     epochs=epochs)
        self.proj_idx = proj_idx

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
        proj_classifier.save_classifier(relative_path=RESULTS, folder=self.folder,
                                        filename=self._set_baseline_filename(seed=self.proj_idx))
        return proj_classifier

    def parallel_train(self, device):
        import multiprocessing
        Parallel(n_jobs=20)(  # multiprocessing.cpu_count()
            delayed(_parallel_train)(dataset_name=self.dataset_name, test=self.test, n_proj=self.n_proj,
                                     proj_idx=proj_idx, size_proj=self.size_proj, proj_mode=self.projection_mode,
                                     device=device)
            for proj_idx in range(self.n_proj))
        self.trained = True


def _set_session(device):
    """ Initialize tf session """
    # print(device_lib.list_local_devices())
    from keras.backend.tensorflow_backend import set_session
    if device == "gpu":
        # print("check cuda: ", tf.test.is_built_with_cuda())
        # print("check gpu: ", tf.test.is_gpu_available())
        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        # elif device == "cpu":
        config.allow_soft_placement = True
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras
        sess.run(tf.global_variables_initializer())
        return sess

# todo: buggy
def _parallel_train(dataset_name, test, n_proj, proj_idx, size_proj, proj_mode, device):
    import tensorflow as tf
    _set_session(device)
    g = tf.Graph()
    with g.as_default():
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)

        model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj,
                                       proj_idx=proj_idx, data_format=data_format, dataset_name=dataset_name,
                                       projection_mode=proj_mode, test=test)
        model.train_single_projection(x_train=x_train, y_train=y_train, device=device, proj_idx=proj_idx)
    del tf


def main(dataset_name, test, proj_idx, size_proj, proj_mode, device):

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)
    model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj, proj_idx=proj_idx,
                           data_format=data_format, dataset_name=dataset_name, projection_mode=proj_mode, test=test)
    model.train_single_projection(x_train, y_train, proj_idx=proj_idx, device=device)

    # ======== buggy =========
    # model.parallel_train(device=device)
    # del model
    # model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj, n_proj=n_proj,
    #                        data_format=data_format, dataset_name=dataset_name, projection_mode=proj_mode,
    #                        test=test)
    # model.load_classifier(relative_path=RESULTS)
    # model.evaluate(x_test, y_test)
    # for attack in ['fgsm','pgd','deepfool','carlini']:
    #     x_test_adv = model.load_adversaries(attack=attack, eps=EPS)
    #     model.evaluate(x_test_adv, y_test)


if __name__ == "__main__":

    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        proj_idx = int(sys.argv[3])
        size_proj_list = list(map(int, sys.argv[4].strip('[]').split(',')))
        projection_mode = sys.argv[5]
        device = sys.argv[6]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        proj_idx = input("\nSet a seed for projections (int): ")
        size_proj_list = input("\nChoose size for the projection (list): ")
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")
        device = input("\nChoose a device (cpu/gpu): ")

    for size_proj in size_proj_list:
        main(dataset_name=dataset_name, test=test, proj_idx=proj_idx, size_proj=size_proj, proj_mode=projection_mode,
             device=device)

