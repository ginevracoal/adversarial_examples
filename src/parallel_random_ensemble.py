# -*- coding: utf-8 -*-

from random_ensemble import *
from projection_functions import compute_single_projection
from utils import load_dataset, _set_session
from joblib import Parallel, delayed

MODEL_NAME = "random_ensemble"
PROJ_MODE = "channels"
DATASETS = "mnist, cifar"


class ParallelRandomEnsemble(RandomEnsemble):

    def __init__(self, input_shape, num_classes, size_proj, proj_idx, n_proj, data_format, dataset_name,
                 projection_mode, library, test, epochs=None, centroid_translation=False):
        super(ParallelRandomEnsemble, self).__init__(input_shape=input_shape, num_classes=num_classes, n_proj=n_proj,
                                                     size_proj=size_proj, projection_mode=projection_mode,
                                                     data_format=data_format, dataset_name=dataset_name, test=test,
                                                     epochs=epochs, centroid_translation=centroid_translation,
                                                     library=library)
        self.proj_idx = proj_idx
        self.n_jobs = 2 if self.test else 20
        # _set_session(device, n_jobs=self.n_jobs)

    @staticmethod
    def _set_session(device):
        K.clear_session()
        return tf.reset_default_graph()

    def train_single_projection(self, x_train, y_train, device, proj_idx):
        """ Trains a single projection of the ensemble classifier and saves the model in current day results folder."""

        print("\nTraining single randens projection with seed =", str(proj_idx),
              "and size_proj =", str(self.size_proj))

        start_time = time.time()

        self.translation_vector = self._set_translation_vector(x_train)
        x_train_projected, x_train_inverse_projected = compute_single_projection(input_data=x_train,
                                                                                 seed=proj_idx,
                                                                                 size_proj=self.size_proj,
                                                                                 projection_mode=self.projection_mode,
                                                                                 translation=self.translation_vector)
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

    def train(self, x_train, y_train, device, n_jobs=2):
        self._set_session(device)
        self.translation_vector = self._set_translation_vector(x_train)
        if self.centroid_translation:
            save_to_pickle(data=self.translation_vector,
                           relative_path="../results/" + str(time.strftime('%Y-%m-%d')) + "/" + self.folder,
                           filename="training_data_centroid.pkl")
        classifiers = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_train)(x_train=x_train, y_train=y_train, dataset_name=self.dataset_name,
                                     input_shape=self.input_shape,
                                     num_classes=self.num_classes, data_format=self.data_format, test=self.test,
                                     proj_idx=proj_idx, device=device, size_proj=self.size_proj,
                                     proj_mode=self.projection_mode, n_jobs=n_jobs,
                                     centroid_translation=self.centroid_translation, library=self.library)
            for proj_idx in range(0,self.n_proj))
        self.trained = True
        self.classifiers = classifiers
        return classifiers

    # def compute_projections(self, input_data):
    #     """
    #     Parallel implementation of this method
    #     :param input_data:
    #     :return:
    #     """
    #     n_jobs = self.n_proj # 2 if device == "gpu" else self.n_proj
    #     projections = Parallel(n_jobs=n_jobs)(
    #         delayed(_parallel_compute_projections)(input_data, proj_idx=proj_idx, size_proj=self.size_proj,
    #                                                projection_mode=self.projection_mode, n_jobs=n_jobs,
    #                                                translation=self.translation)
    #         for proj_idx in self.random_seeds)
    #
    #     # eventually adjust input dimension to a single channel projection
    #     projections = np.array(projections)
    #
    #     if projections.shape[4] == 1:
    #         self.input_shape = (self.input_shape[0], self.input_shape[1], 1)
    #     else:
    #         self.input_shape = (self.input_shape[0], self.input_shape[1], 3)
    #     return projections, None

    def _sum_ensemble_classifier(self, classifiers, projected_data):
        """
        Parallelized version of this method.
        :param classifiers: list of trained classifiers
        :param projected_data: list of projected data for all of the n_proj random initializations
        :return:
        """
        # compute predictions for each projection
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_predict)(classifier=classifier, projected_data=projected_data[i], device=device,
                                       n_jobs=self.n_jobs)
            for i, classifier in enumerate(classifiers))
        proj_predictions = np.array(results)
        # sum the probabilities across all predictors
        predictions = np.sum(proj_predictions, axis=0)
        return predictions

    def load_classifier(self, relative_path, folder=None, filename=None):
        n_jobs = self.n_proj
        K.clear_session()
        self._set_session(device)
        start_time = time.time()
        classifiers = Parallel(n_jobs=n_jobs)(
            delayed(_parallel_load_classifier)(input_shape = self.input_shape, num_classes=self.num_classes,
                                               data_format=self.data_format, dataset_name=self.dataset_name,
                                               relative_path=relative_path, folder=self.folder, n_jobs=n_jobs,
                                               filename=self._set_baseline_filename(seed=i))
            for i in list(self.random_seeds))
        print("\nLoading time: --- %s seconds ---" % (time.time() - start_time))
        # for classifier in classifiers:
        #     classifier.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
        #                            metrics=['accuracy'])
        #     # classifier.model._make_predict_function()
        self.classifiers = classifiers
        if self.centroid_translation:
            self.translation_vector = load_from_pickle(path=relative_path + self.folder + "training_data_centroid.pkl",
                                                       test=False)
        return classifiers

    def evaluate(self, x, y, report_projections=False, model_path=RESULTS, device="cpu", add_baseline_prob=False):
        """
        Computes parallel evaluation of the model, then joins the results from the single workers into the final
        probability vector.
        :param x: Input data
        :param y: Input labels
        :param report_projections: include classification labels (True/False)
        :param model_path: model path for loading (RESULTS/TRAINED_MODELS)
        :param device: model evaluation device (True/False)
        :return: accuracy on the predictions
        """
        K.clear_session()
        self._set_session(device)
        if self.centroid_translation:
            translation = load_from_pickle(path=model_path + self.folder + "training_data_centroid.pkl",
                                                       test=False)
        else:
            translation = None

        proj_predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_evaluate)(input_shape=self.input_shape, num_classes=self.num_classes, test=self.test,
                                        data_format=self.data_format, dataset_name=self.dataset_name,
                                        relative_path=model_path, folder=self.folder, input_data=x, seed=i,
                                        filename=self._set_baseline_filename(seed=i), size_proj=self.size_proj,
                                        projection_mode=self.projection_mode, translation=translation)
            for i in list(self.random_seeds))
        predictions = np.sum(np.array(proj_predictions), axis=0)
        # print(np.array(proj_predictions)[:,0])
        # print(predictions[0])
        # exit()

        if add_baseline_prob:
            print("\nAdding baseline probability vector to the predictions.")
            baseline = BaselineConvnet(input_shape=self.original_input_shape, num_classes=self.num_classes,
                                       data_format=self.data_format, dataset_name=self.dataset_name, test=self.test)
            baseline.load_classifier(relative_path=TRAINED_MODELS, folder="baseline/", filename=baseline.filename)
            baseline_predictions = baseline.predict(x)

            # sum the probabilities across all predictors
            predictions = np.add(predictions, baseline_predictions)

        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y, axis=1)
        nb_correct_adv_pred = np.sum(y_pred == y_true)

        print("\nCorrectly classified: {}".format(nb_correct_adv_pred))
        print("Incorrectly classified: {}".format(len(x) - nb_correct_adv_pred))

        acc = nb_correct_adv_pred / y.shape[0]
        print("Accuracy: %.2f%%" % (acc * 100))
        return acc

def _parallel_evaluate(input_shape, num_classes, test, data_format, dataset_name, relative_path, folder, filename,
                       size_proj, input_data, seed, projection_mode, translation):
    """ Parallel evaluation on single projections using BaselineConvnet base class. """
    classifier = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, test=test, data_format=data_format,
                                 dataset_name=dataset_name)
    classifier.load_classifier(relative_path=relative_path, folder=folder, filename=filename)
    print("Parallel computing projection ", seed)
    projection, _ = compute_single_projection(input_data=input_data, seed=seed, size_proj=size_proj,
                                              projection_mode=projection_mode, translation=translation)
    prediction = classifier.predict(projection)
    return prediction


def _parallel_predict(classifier, projected_data, n_jobs, device):
    # import tensorflow as tf
    # _set_session(device, n_jobs)
    # use the same computational graph of training for the predictions
    # g = tf.get_default_graph()
    # with g.as_default():
    predictions = classifier.predict(projected_data)
    # del tf
    # del g
    return predictions


def _parallel_train(x_train, y_train, input_shape, num_classes, data_format, dataset_name, test, proj_idx, size_proj,
                    proj_mode, n_jobs, device, centroid_translation, library):
    print("\nParallel training projection ", proj_idx)
    # import tensorflow as tf
    # g = tf.get_default_graph()
    # _set_session(device, n_jobs)
    # with g.as_default():
    model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj,
                                   proj_idx=proj_idx, data_format=data_format, dataset_name=dataset_name,
                                   projection_mode=proj_mode, test=test, n_proj=1,
                                   centroid_translation=centroid_translation, library=library)
    model.train_single_projection(x_train=x_train, y_train=y_train, device=device, proj_idx=proj_idx)
    # del tf
    # del g

#
# def _parallel_compute_projections(input_data, proj_idx, size_proj, projection_mode, n_jobs, translation):
#     _set_session(device, n_jobs)
#     print("\nParallel computing projection ", proj_idx)
#     projection, _ = compute_single_projection(input_data=input_data, seed=proj_idx, size_proj=size_proj,
#                                               projection_mode=projection_mode, translation=translation)
#     return projection


def _parallel_load_classifier(input_shape, num_classes, data_format, dataset_name, relative_path, folder, filename,
                              n_jobs):
    # _set_session(device, n_jobs)
    classifier = BaselineConvnet(input_shape=input_shape, num_classes=num_classes,
                                 test=test, data_format=data_format, dataset_name=dataset_name)
    classifier.load_classifier(relative_path=relative_path, folder=folder, filename=filename)
    return classifier


def main(dataset_name, test, n_proj, size_proj, proj_mode, device):

    attacks = ["fgsm","pgd","deepfool", "carlini"]

    seed=0
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)

    baseline = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                               dataset_name=dataset_name, test=test, epochs=None, library="cleverhans")
    baseline.load_classifier(relative_path=TRAINED_MODELS, filename=baseline.filename+"_seed="+str(seed))

    # === parallel train, save and load the whole ensemble === #
    model = ParallelRandomEnsemble(input_shape=input_shape, num_classes=num_classes, size_proj=size_proj,
                                   proj_idx=None, n_proj=n_proj, data_format=data_format, dataset_name=dataset_name,
                                   projection_mode=proj_mode, test=test, centroid_translation=False,
                                   library="cleverhans")
    # model.train(x_train, y_train, device=device)
    model_path = TRAINED_MODELS
    model.load_classifier(relative_path=model_path)

    add_baseline_prob=False
    print("\n== test set ==")
    model.evaluate(x=x_test, y=y_test, device=device, model_path=model_path, add_baseline_prob=add_baseline_prob)
    for attack in attacks:
        print("\n== "+str(attack)+" attack ==")
        x_test_adv = model.load_adversaries(attack=attack, relative_path=RESULTS)
        model.evaluate(x_test_adv, y_test, device=device, model_path=model_path, add_baseline_prob=add_baseline_prob)


if __name__ == "__main__":

    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        n_proj = int(sys.argv[3])
        size_proj_list = list(map(int, sys.argv[4].strip('[]').split(',')))
        projection_mode = sys.argv[5]
        device = sys.argv[6]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        n_proj = input("\nChoose the number of projections (type=int): ")
        size_proj_list = input("\nChoose size for the projection (list): ")
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")
        device = input("\nChoose a device (cpu/gpu): ")

    for size_proj in size_proj_list:
        main(dataset_name=dataset_name, test=test, n_proj=n_proj, size_proj=size_proj,
             proj_mode=projection_mode, device=device)

