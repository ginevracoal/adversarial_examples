# -*- coding: utf-8 -*-

"""
This model computes random projections of the input points in a lower dimensional space and performs classification
separately on each projection, then it returns an ensemble classification on the original input data.
"""

from adversarial_classifier import *
from baseline_convnet import BaselineConvnet
import sys
from projection_functions import *

REPORT_PROJECTIONS = False
ADD_BASELINE_PROB = False
MODEL_NAME = "random_ensemble"
TRAINED_MODELS = "../trained_models/random_ensemble/"
PROJ_MODE = "flat, channels, one_channel, grayscale"


class RandomEnsemble(BaselineConvnet):
    """
    Classifies `n_proj` random projections of the training data in a lower dimensional space (whose dimension is
    `size_proj`^2), then classifies the original high dimensional data with an ensemble classifier, summing up the
    probabilities from the single projections.
    """
    def __init__(self, input_shape, num_classes, n_proj, size_proj, projection_mode, data_format, dataset_name, test):
        """
        Extends BaselineConvnet initializer with additional informations about the projections.

        :param input_shape:
        :param num_classes:
        :param n_proj: number of random projections
        :param size_proj: size of a random projection
        :param projection_mode: method for computing projections on RGB images
        :param data_format: channels first or last
        :param dataset_name:
        :param test: if True only takes the first 100 samples
        """

        if size_proj > input_shape[1]:
            raise ValueError("The size of projections has to be lower than the image size.")

        super(RandomEnsemble, self).__init__(input_shape, num_classes, data_format, dataset_name, test)
        self.input_shape = (size_proj, size_proj, input_shape[2])
        self.n_proj = n_proj
        self.size_proj = size_proj
        self.projection_mode = projection_mode
        self.random_seeds = np.array([123, 45, 180, 172, 61, 63, 70, 83, 115, 67, 56, 133, 12, 198, 156,
                                      54, 42, 150, 184, 52, 17, 127, 13])
        self.trained = False
        self.training_time = 0
        self.ensemble_method = "sum"  # supported methods: mode, sum
        self.x_test_proj = None
        self.baseline_classifier = None

        print("\n === RandEns model ( n_proj = ", self.n_proj, ", size_proj = ", self.size_proj, ") ===")

    def compute_projections(self, input_data):
        """ Extends utils.compute_projections method in order to handle the third input dimension."""
        projections, inverse_projections = compute_projections(input_data, self.random_seeds, self.n_proj,
                                                               self.size_proj, self.projection_mode)

        # eventually adjust input dimension to a single channel projection
        if projections.shape[4] == 1:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 1)
        else:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 3)

        # sess = tf.Session()
        # sess.as_default()
        # projections = projections.eval(session=sess)
        # inverse_projections = inverse_projections.eval(session=sess)
        return projections, inverse_projections

    def train(self, x_train, y_train, batch_size, epochs):
        """
        Trains the baseline model over `n_proj` random projections of the training data whose input shape is
        `(size_proj, size_proj, 1)`.

        :param x_train:
        :param y_train:
        :param batch_size:
        :param epochs:
        :return: list of n_proj trained models, which are art.KerasClassifier fitted objects
        """

        start_time = time.time()
        input_data = x_train.astype(float)
        x_train_projected, _ = self.compute_projections(input_data=input_data)

        # # eventually adjust input dimension to a single channel projection
        # if x_train_projected.shape[4] == 1:
        #     self.input_shape = (self.input_shape[0],self.input_shape[1],1)

        classifiers = []
        for i in range(self.n_proj):
            # use the same model architecture (not weights) for all trainings
            baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                       data_format=self.data_format, dataset_name=self.dataset_name, test=self.test)
            # train n_proj classifiers on different training data
            classifiers.append(baseline.train(x_train_projected[i], y_train, batch_size=batch_size, epochs=epochs))
            del baseline

        print("\nTraining time for model ( n_proj =", str(self.n_proj), ", size_proj =", str(self.size_proj),
              "): --- %s seconds ---" % (time.time() - start_time))

        self.trained = True
        return classifiers

    @staticmethod
    def _sum_ensemble_classifier(classifiers, projected_data):
        """
        :param classifiers: list of `n_proj` different GaussianRandomProjection objects
        :param projected_data: array of test data projected on the different `n_proj` training random directions
        (`size_proj` directions for each projection)
        :return: sum of all the predicted probabilities among each class for the `n_proj` classifiers

        ```
        Predictions on the first element by each single classifier
        predictions[:, 0] =
         [[0.04745529 0.00188083 0.01035858 0.21188359 0.00125252 0.44483757
          0.00074033 0.13916749 0.01394993 0.1284739 ]
         [0.00259137 0.00002327 0.48114488 0.42658636 0.00003032 0.01012747
          0.00002206 0.03735029 0.02623402 0.0158899 ]
         [0.00000004 0.00000041 0.00000277 0.00001737 0.00000067 0.00000228
          0.         0.9995009  0.00000014 0.0004754 ]]

        Ensemble prediction vector on the first element
        summed_predictions[0] =
         [0.05004669 0.00190451 0.49150622 0.6384873  0.0012835  0.45496735
         0.0007624  1.1760187  0.04018409 0.14483918]
        """
        # compute predictions for each projection
        proj_predictions = np.array([classifier.predict(projected_data[i]) for i, classifier in enumerate(classifiers)])
        # sum the probabilities across all predictors
        predictions = np.sum(proj_predictions, axis=0)
        return predictions

    @staticmethod
    def _mode_ensemble_classifier(classifiers, projected_data):
        """
        :param classifiers: list of `n_proj` different GaussianRandomProjection objects
        :param projected_data: array of test data projected on the different `n_proj` training random directions
        (`size_proj` directions for each projection)
        :return: compute the argmax of the probability vectors and then, for each points, choose the mode over all
        classes as the predicted label

        ```
        Computing random projections.
        Input shape:  (100, 28, 28, 1)
        Projected data shape: (3, 100, 8, 8, 1)

        Predictions on the first element by each single classifier
        predictions[:, 0] =
        [[0.09603461 0.08185963 0.03264992 0.07047556 0.2478332  0.03418195
          0.13880958 0.19712913 0.04649669 0.05452974]
         [0.0687536  0.14464664 0.0766349  0.09082405 0.1066305  0.01555605
          0.03265413 0.12625733 0.14203466 0.19600812]
         [0.16379683 0.0895557  0.07057846 0.09945401 0.25141633 0.04555665
          0.08481159 0.0610559  0.06158736 0.07218721]]

        argmax_predictions[:, 0] =
        [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
         [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]

        Ensemble prediction vector on the first element
        summed_predictions[0] =
        [0.         0.         0.         0.         0.66666667 0.
         0.         0.         0.         0.33333333]
        ```
        """

        # compute predictions for each projection
        proj_predictions = np.array([classifier.predict(projected_data[i]) for i, classifier in enumerate(classifiers)])

        # convert probability vectors into target vectors
        argmax_predictions = np.zeros(proj_predictions.shape)
        for i in range(proj_predictions.shape[0]):
            idx = np.argmax(proj_predictions[i], axis=-1)
            argmax_predictions[i, np.arange(proj_predictions.shape[1]), idx] = 1
        # sum the probabilities across all predictors
        predictions = np.sum(argmax_predictions, axis=0)
        # normalize
        predictions = predictions / predictions.sum(axis=1)[:, None]

        return predictions

    def predict(self, classifiers, data, add_baseline_prob=ADD_BASELINE_PROB, *args, **kwargs):
        """
        Compute the ensemble prediction.

        :param classifiers: list of trained classifiers over different projections
        :param data: input data
        :param add_baseline_prob: if True adds baseline probabilities to logits layer
        :return: final predictions for the input data
        """
        projected_data, _ = self.compute_projections(data)

        predictions = None
        if self.ensemble_method == 'sum':
            predictions = self._sum_ensemble_classifier(classifiers, projected_data)
        elif self.ensemble_method == 'mode':
            predictions = self._mode_ensemble_classifier(classifiers, projected_data)

        if add_baseline_prob:
            baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                       data_format=self.data_format, dataset_name=self.dataset_name, test=self.test)
            if self.baseline_classifier is None:
                self.baseline_classifier = baseline.load_classifier(relative_path="../trained_models/baseline/")

            baseline_predictions = np.array(baseline.predict(self.baseline_classifier, data))
            # sum the probabilities across all predictors
            final_predictions = np.add(predictions, baseline_predictions)
            return final_predictions
        else:
            return predictions

    def report_projections(self, classifier, x_test_proj, y_test):
        """
        Computes classification reports on each projection.
        """
        print("\n === projections report ===")
        baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                   data_format=self.data_format, dataset_name=self.dataset_name, test=True)
        for i, proj_classifier in enumerate(classifier):
            print("\nTest evaluation on projection ", self.random_seeds[i])
            baseline.evaluate(classifier=proj_classifier, x=x_test_proj[i], y=y_test)

    def evaluate(self, classifier, x, y, report_projections=REPORT_PROJECTIONS):
        """ Extends evaluate() with projections reports"""
        y_pred = super(RandomEnsemble, self).evaluate(classifier, x, y)
        if report_projections:
            self.x_proj, _ = self.compute_projections(x)
            self.report_projections(classifier=classifier, x_test_proj=self.x_proj, y_test=y)

        return y_pred

    # def evaluate_test(self, classifier, x_test, y_test, report_projections=REPORT_PROJECTIONS):
    #     """ Extends evaluate_test() with projections reports"""
    #     y_test_pred = super(RandomEnsemble, self).evaluate(classifier, x_test, y_test)
    #     if report_projections:
    #
    #         self.x_test_proj, _ = self.compute_projections(x_test)
    #         self.report_projections(classifier=classifier, x_test_proj=self.x_test_proj, y_test=y_test)
    #
    #     return y_test_pred
    #
    # def evaluate_adversaries(self, classifier, x_test, y_test, method, dataset_name, adversaries_path=None, test=False,
    #                          report_projections=REPORT_PROJECTIONS):
    #     """ Extends evaluate_adversaries() with projections reports"""
    #     x_test_adv, y_test_adv = super(RandomEnsemble, self).evaluate(classifier=classifier, x=x_test, y=y_test)
    #     if report_projections:
    #         x_test_adv_proj, _ = self.compute_projections(x_test_adv)
    #         self.report_projections(classifier=classifier, x_test_proj=x_test_adv_proj, y_test=y_test)

    def generate_adversaries(self, classifier, x, y, method, dataset_name, adversaries_path=None, test=False, *args, **kwargs):
        # todo: refactor
        """ Adversaries are generated on the baseline classifier """

        baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                   data_format=self.data_format, dataset_name=self.dataset_name, test=self.test)
        if self.baseline_classifier is None and adversaries_path is None:
            self.baseline_classifier = baseline.load_classifier(relative_path="../trained_models/baseline/")
        x_adv = baseline.generate_adversaries(classifier=self.baseline_classifier, x=x, y=y, method=method,
                                              dataset_name=dataset_name, test=test, eps=eps)
        return x_adv

    def save_classifier(self, classifier, model_name=MODEL_NAME):
        """
        Saves all projections classifiers separately.
        :param classifier: list of projection classifiers
        :param model_name: name of the model
        """
        # todo: salvare il modello soltanto nel caso n_proj=15. Per le valutazioni su n_proj inferiori basta il loading corretto.
        if self.trained:
            for i, proj_classifier in enumerate(classifier):
                filename = model_name + "_size=" + str(self.size_proj) + "_" + str(self.random_seeds[i])+".h5"
                folder = str(self.dataset_name) + "_" + model_name + "_size=" + str(self.size_proj) + "_epochs=" + \
                         str(self.epochs) + "_" + str(self.projection_mode) + "/"
                proj_classifier.save(filename=filename,
                                     path=RESULTS + time.strftime('%Y-%m-%d') + "/" + folder )
        else:
            raise ValueError("Model has not been fitted!")

    def load_classifier(self, relative_path):
        """
        Loads a pretrained classifier and sets the projector with the training seed.
        :param relative_path: relative path of the folder containing the list of trained classifiers
        :return: list of trained classifiers
        """
        start_time = time.time()
        # _, epochs = self._set_training_params(dataset_name=self.dataset_name,test=False).values() # load full epochs model

        # old
        # folder = relative_path + str(self.dataset_name) + "_" + MODEL_NAME + "_size=" + str(self.size_proj) + \
        #          "_epochs=" + str(self.epochs) + "_" + str(self.projection_mode)+ "/"
        # trained_models = [load_model(folder +
        #                              MODEL_NAME + "_size=" + str(self.size_proj) + "_" + str(seed) + ".h5")
        #                   for seed in self.random_seeds[:self.n_proj]]
        # print("\nLoading time: --- %s seconds ---" % (time.time() - start_time))
        #
        #
        # classifiers = [KerasClassifier((MIN, MAX), model, use_logits=False) for model in trained_models]

        # new
        paths = [relative_path + str(self.dataset_name) + "_" + MODEL_NAME + "_size=" + str(self.size_proj) + \
                 "_epochs=" + str(self.epochs) + "_" + str(self.projection_mode)+ "/" + MODEL_NAME + "_size=" +
                 str(self.size_proj) + "_" + str(seed) + ".h5" for seed in self.random_seeds[:self.n_proj]]

        classifiers = [super(BaselineConvnet, self).load_classifier(path) for path in paths]
        print("\nLoading time: --- %s seconds ---" % (time.time() - start_time))

        return classifiers

    def load_robust_classifier(self, relative_path, attack, eps):
        raise NotImplementedError


########
# MAIN #
########


def main(dataset_name, test, n_proj, size_proj, projection_mode, attack, eps):
    """
    :param dataset: choose between "mnist" and "cifar"
    :param test: if True only takes 100 samples
    :param n_proj: number of projections. Trained models used values: 6, 9, 12, 15.
    :param size_proj: size of each projection. You can currently load models with values: 8, 12, 16, 20.
    :param projection_mode: method for computing projections on RGB images
    :param attack: attack name
    :param eps: max norm of a perturbation
    """

    # === initialize === #
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)

    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes,
                           n_proj=n_proj, size_proj=size_proj, projection_mode=projection_mode,
                           data_format=data_format, dataset_name=dataset_name, test=test)
    # === train === #
    # classifier = model.train(x_train, y_train, batch_size=model.batch_size, epochs=model.epochs)
    # model.save_classifier(classifier=classifier, model_name=MODEL_NAME)

    # === load classifier === #
    # classifier = model.load_classifier(relative_path=TRAINED_MODELS)

    # === evaluate === #
    # model.evaluate(classifier=classifier, x=x_test, y=y_test)

    # x_test_adv = model.load_adversaries(dataset_name=dataset_name,attack=attack, eps=eps, test=test)
    # print("Distance from perturbations: ", compute_distances(x_test, x_test_adv, ord=model._get_norm(attack)))
    # model.evaluate(classifier=classifier, x=x_test_adv, y=y_test)

    # === generate perturbations === #
    # compute_variances(x_test, y_test)
    projections, inverse_projections = model.compute_projections(input_data=x_test)
    perturbations, augmented_inputs = compute_perturbations(input_data=x_test, inverse_projections=inverse_projections)

    # # print(np.mean([compute_angle(x_test[i],augmented_inputs[i]) for i in range(len(x_test))]))
    # # exit()
    # eig_vals, eig_vecs = compute_linear_discriminants(x_test, y_test)
    # print(eig_vals,"\n",eig_vecs)
    # exit()
    # # print(x_test[0,0,0,:],augmented_inputs[0,0,0,:],"\n")
    # avg_distance = lambda x: np.mean([np.linalg.norm(x[0][idx]-x[1][idx]) for idx in range(len(x_test))])
    # print("Average distance from attack: ", avg_distance([x_test, augmented_inputs]))

    # # # === evaluate baseline on perturbations === #
    # baseline = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
    #                         dataset_name=dataset_name, test=True)
    # rel_path = "../trained_models/baseline/" + str(dataset_name) + "_baseline.h5"
    # baseline_classifier = baseline.load_classifier(relative_path=rel_path)
    # baseline.evaluate(baseline_classifier, augmented_inputs, y_test)

    # === plot perturbations === #
    # plot_images(image_data_list=[x_test, projections[0], inverse_projections[0]])
    plot_images(image_data_list=[x_test,perturbations,augmented_inputs])

if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        n_proj_list = list(map(int, sys.argv[3].strip('[]').split(',')))
        size_proj_list = list(map(int, sys.argv[4].strip('[]').split(',')))
        projection_mode = sys.argv[5]
        attack = sys.argv[6]
        eps = float(sys.argv[7])

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        n_proj_list = list(map(int, input("\nChoose the number of projections (type=list): ").strip('[]').split(',')))
        size_proj_list = list(map(int, input("\nChoose the size of projections (type=list): ").strip('[]').split(',')))
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")
        attack = input("\nChoose an attack ("+ATTACKS+"): ")
        eps = float(input("\nSet a ths for perturbation norm: "))


    for n_proj in n_proj_list:
        for size_proj in size_proj_list:
            K.clear_session()
            main(dataset_name=dataset_name, test=test, n_proj=n_proj, size_proj=size_proj,
                 projection_mode=projection_mode, attack=attack, eps=eps)

