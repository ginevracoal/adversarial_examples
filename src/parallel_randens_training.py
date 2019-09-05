from random_ensemble import *
import sys
import multiprocessing as mp
import logging
from baseline_convnet import BaselineConvnet
from projection_functions import compute_single_projection

MODEL_NAME = "random_ensemble"


class ParallelRandomEnsemble(RandomEnsemble):

    def __init__(self, input_shape, num_classes, size_proj, data_format, dataset_name, projection_mode):
        super(ParallelRandomEnsemble, self).__init__(input_shape, num_classes, None, size_proj, projection_mode,
                                                     data_format, dataset_name, test=False)
        # None refers to n_proj since we only have to compute a single projection

    def train_single_projection(self, x_train, y_train, batch_size, epochs, idx, save):
        """ Trains a single projection of the ensemble classifier and saves the model in current day results folder."""
        K.clear_session()

        start_time = time.time()
        x_train_projected, x_train_inverse_projected = compute_single_projection(input_data=x_train,
                                                                                 seed=self.random_seeds[idx],
                                                                                 size_proj=self.size_proj,
                                                                                 projection_mode=self.projection_mode)

        # eventually adjust input dimension to a single channel projection
        if x_train_projected.shape[3] == 1:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 1)

        # use the same model architecture (not weights) for all trainings
        baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                   dataset_name=self.dataset_name, data_format=self.data_format, test=False)
        classifier = baseline.train(x_train_projected, y_train, batch_size=batch_size, epochs=epochs)
        print("\nTraining time for single projection with size_proj=", str(self.size_proj),
              "): --- %s seconds ---" % (time.time() - start_time))

        if save:
            start = time.time()
            filename = MODEL_NAME + "_size=" + str(self.size_proj) + "_" + str(self.random_seeds[idx]) + ".h5"
            folder = str(self.dataset_name) + "_" + MODEL_NAME + "_sum_size=" + str(self.size_proj) + "_" + \
                     str(self.projection_mode) + "/"
            classifier.save(filename=filename, path=RESULTS + time.strftime('%Y-%m-%d') + "/" + folder)
            saving_time = time.time() - start
            self.training_time -= saving_time

        return classifier

    # todo: buggy
    # def parallel_train(self, x_train, y_train, batch_size, epochs):
    #     """
    #     Trains the baseline model over `n_proj` random projections of the training data whose input shape is
    #     `(size_proj, size_proj, 1)` and parallelizes training over the different projections.
    #
    #     :param x_train:
    #     :param y_train:
    #     :param batch_size:
    #     :param epochs:
    #     :return: list of n_proj trained models, which are art.KerasClassifier fitted objects
    #     """
    #     mpl = mp.log_to_stderr()
    #     mpl.setLevel(logging.INFO)
    #
    #     start = time.time()
    #     x_train_projected, _ = compute_projections(x_train, random_seeds=self.random_seeds,
    #                                             n_proj=self.n_proj, size_proj=self.size_proj)
    #
    #     # Define an output queue
    #     #output = mp.Queue()
    #     # Setup a list of processes that we want to run
    #     processes = [mp.Process(target=self.train_single_projection,
    #                             args=(x_train, y_train, batch_size, epochs, i, True)) for i in range(self.n_proj)]
    #     # Run processes
    #     for p in processes:
    #         p.start()
    #
    #     # Exit the completed processes
    #     for p in processes:
    #         p.join()
    #         #output.close()
    #
    #     self.training_time += time.time() - start
    #
    #     # Get process results from the output queue
    #     #classifiers = [output.get() for p in processes]
    #
    #     print("\nParallel training time for model ( n_proj =", str(self.n_proj), ", size_proj =", str(self.size_proj),
    #           "): --- %s seconds ---" % (self.training_time))
    #     classifiers = self.load_classifier(relative_path=RESULTS+time.strftime('%Y-%m-%d')+"/",
    #                                        model_name="random_ensemble")
    #
    #     self.trained = True
    #
    #     return classifiers


def main(dataset_name, test, proj_idx, size_proj, proj_mode):

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)

    model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj,
                                   data_format=data_format, dataset_name=dataset_name, projection_mode=proj_mode)
    model.train_single_projection(x_train=x_train, y_train=y_train, batch_size=model.batch_size,
                                  epochs=model.epochs, idx=proj_idx, save=True)


if __name__ == "__main__":

    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        proj_idx = int(sys.argv[3])
        size_proj_list = list(map(int, sys.argv[4].strip('[]').split(',')))
        projection_mode = sys.argv[5]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        proj_idx = input("\nChoose the projection idx. ")
        size_proj_list = input("\nChoose size for the projection. ")
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+")")

    for size_proj in size_proj_list:
        K.clear_session()
        main(dataset_name=dataset_name, test=test, proj_idx=proj_idx, size_proj=size_proj, proj_mode=projection_mode)


