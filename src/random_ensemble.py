# -*- coding: utf-8 -*-

"""
This model computes random projections of the input points in a lower dimensional space and performs classification
separately on each projection, then it returns an ensemble classification on the original input data.
"""

from baseline_convnet import *
from projection_functions import *
from robustness_measures import softmax_difference

############
# defaults #
############

REPORT_PROJECTIONS = False
PROJ_MODE = "flat, channels, one_channel, grayscale"
MODEL_NAME = "randens"


class RandomEnsemble(BaselineConvnet):
    """
    Classifies `n_proj` random projections of the training data in a lower dimensional space (whose dimension is
    `size_proj`^2), then classifies the original high dimensional data with an ensemble classifier, summing up the
    probabilities from the single projections.
    """
    def __init__(self, input_shape, num_classes, n_proj, size_proj, projection_mode, data_format, dataset_name, test,
                 library, epochs=None, centroid_translation=False):
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
        # todo: refactor centroid translation
        self.centroid_translation = centroid_translation
        self.translation_vector = None
        ######################################
        self.original_input_shape = input_shape
        self.n_proj = n_proj
        self.size_proj = size_proj
        self.projection_mode = projection_mode
        super(RandomEnsemble, self).__init__(input_shape, num_classes, data_format, dataset_name, test, library, epochs)
        self.random_seeds = range(0,n_proj)  # random.sample(list(range(1, 1000)), n_proj)
        # self.random_seeds = np.array([123, 45, 180, 172, 61, 63, 70, 83, 115, 67, 56, 133, 12, 198, 156,)
        self.random_seeds = range(0,n_proj)  # random.sample(list(range(1, 1000)), n_proj)
        # self.random_seeds = np.array([123, 45, 180, 172, 61, 63, 70, 83, 115, 67, 56, 133, 12, 198, 156,
        #                               54, 42, 150, 184, 52, 17, 127, 13])
        self.input_shape = (size_proj, size_proj, input_shape[2])
        self.classifiers = None
        self.training_time = 0
        self.ensemble_method = "sum"  # supported methods: mode, sum
        self.x_test_proj = None
        self.baseline_classifier = None
        print("\n === RandEns model ( n_proj = ", self.n_proj, ", size_proj = ", self.size_proj, ") ===")

    def compute_projections(self, input_data, translation):
        """ Extends utils.compute_projections method in order to handle the third input dimension."""
        projections, inverse_projections = compute_projections(input_data, self.random_seeds, self.n_proj,
                                                               self.size_proj, self.projection_mode,
                                                               translation=translation)

        # eventually adjust input dimension to a single channel projection
        if projections.shape[4] == 1:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 1)
        else:
            self.input_shape = (self.input_shape[0], self.input_shape[1], 3)

        return projections, inverse_projections

    def _set_translation_vector(self, x_train):
        if self.centroid_translation:
            return np.mean(np.mean(x_train, axis=0), axis=2)
        else:
            return None
            # return np.zeros(shape=[1, rows, cols, channels], dtype=np.float32)

    def train(self, x_train, y_train, device):
        """
        Trains the baseline model over `n_proj` random projections of the training data whose input shape is
        `(size_proj, size_proj, 1)`.

        :param x_train: training data
        :param y_train: training labels
        :return: list of n_proj trained models, which are art.KerasClassifier fitted objects
        """
        self.translation_vector = self._set_translation_vector(x_train)
        device_name = self._set_device_name(device)

        with tf.device(device_name):
            start_time = time.time()
            input_data = x_train.astype(float)
            x_train_projected, _ = self.compute_projections(input_data=input_data,
                                                            translation=self.translation_vector)

            # eventually adjust input dimension to a single channel projection
            if x_train_projected.shape[4] == 1:
                self.input_shape = (self.input_shape[0],self.input_shape[1],1)

            classifiers = []
            for i in self.random_seeds:
                # use the same model architecture (not weights) for all trainings
                baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                           data_format=self.data_format, dataset_name=self.dataset_name, test=self.test)
                # train n_proj classifiers on different training data
                classifiers.append(baseline.train(x_train_projected[i], y_train, device))
                del baseline

            print("\nTraining time for model ( n_proj =", str(self.n_proj), ", size_proj =", str(self.size_proj),
                  "): --- %s seconds ---" % (time.time() - start_time))

            self.trained = True
            self.classifiers = classifiers
            return classifiers

    def _sum_ensemble_classifier(self, classifiers, projected_data):
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

        # todo: little analysis on ensemble behaviour
        # print(proj_predictions[:,0,:])

        # sum the probabilities across all predictors
        predictions = np.sum(proj_predictions, axis=0)
        return predictions

    def _mode_ensemble_classifier(self, classifiers, projected_data):
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

    def predict(self, x, add_baseline_prob=False):
        """
        Compute the ensemble prediction probability vector by summing up the probability vectors obtained over different
        projections.
        :param x: input data
        :param add_baseline_prob: if True adds baseline probabilities to logits layer
        :return: probability vector final predictions on x
        """
        projected_data, _ = self.compute_projections(x, translation=self.translation_vector)
        predictions = None
        if self.ensemble_method == 'sum':
            predictions = self._sum_ensemble_classifier(self.classifiers, projected_data)
        elif self.ensemble_method == 'mode':
            predictions = self._mode_ensemble_classifier(self.classifiers, projected_data)

        if add_baseline_prob:
            print("\nAdding baseline probability vector to the predictions.")
            baseline = BaselineConvnet(input_shape=self.original_input_shape, num_classes=self.num_classes,
                                       data_format=self.data_format, dataset_name=self.dataset_name, test=self.test)
            baseline.load_classifier(relative_path=TRAINED_MODELS, folder="baseline/", filename=baseline.filename)
            baseline_predictions = baseline.predict(x)

            # sum the probabilities across all predictors
            final_predictions = np.add(predictions, baseline_predictions)
            return final_predictions
        else:
            return predictions

    def report_projections(self, classifiers, x_test_proj, y_test):
        """
        Computes classification reports on each projection.
        """
        print("\n === projections report ===")
        # proj_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
        #                            data_format=self.data_format, dataset_name=self.dataset_name, test=True)
        for i, proj_classifier in enumerate(classifiers):
            print("\nTest evaluation on projection ", self.random_seeds[i])
            proj_classifier.evaluate(x=x_test_proj[i], y=y_test)

    def evaluate(self, x, y, report_projections=False):
        """ Extends evaluate() with projections reports"""
        classification_prob, y_true, y_pred = super(RandomEnsemble, self).evaluate(x, y)
        # y_pred = self.classifiers.evaluate(x, y)
        if report_projections:
            x_proj, _ = self.compute_projections(x, translation=self.translation_vector)
            self.report_projections(classifiers=self.classifiers, x_test_proj=x_proj, y_test=y)
        # print(classification_prob[0],y_true[0],y_pred[0])
        return classification_prob, y_true, y_pred

    def generate_adversaries(self, x, y, attack, seed=0, eps=None, device="cpu"):
        """ Adversaries are generated on the baseline classifier """

        baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                   data_format=self.data_format, dataset_name=self.dataset_name, test=self.test)
        baseline.load_classifier(relative_path=TRAINED_MODELS + MODEL_NAME + "/")
        x_adv = baseline.generate_adversaries(x, y, attack=attack, eps=eps)
        return x_adv

    def _set_model_path(self, model_name="randens"):
        folder = MODEL_NAME + "/" + self.dataset_name + "_" + str(model_name) + "_size=" + str(self.size_proj) +\
                 "_" + str(self.projection_mode)
        if self.centroid_translation:
            folder = folder + "_centroid"
        return {'folder': folder + "/", 'filename': None}

    def _set_baseline_filename(self, seed):
        """ Sets baseline filenames inside randens folder based on the projection seed. """
        filename = self.dataset_name + "_baseline" + "_size=" + str(self.size_proj) + \
                   "_" + str(self.projection_mode)
        if self.centroid_translation:
            filename = filename + "_centroid"

        if self.epochs == None:
            filename = filename + "_" + str(seed)
        else:
            filename = filename + "_epochs=" + str(self.epochs) + "_" + str(seed)
        return filename

    def save_classifier(self, relative_path, folder=None, filename=None):
        """
        Saves all projections classifiers separately.
        :param relative_path: relative path of the folder containing the list of trained classifiers.
                              It can be either TRAINED_MODELS or RESULTS
        :param filename: filename
        """
        if self.trained:
            for i, proj_classifier in enumerate(self.classifiers):
                proj_classifier.save_classifier(relative_path=relative_path, folder=self.folder,
                                                filename=self._set_baseline_filename(seed=i))
            if self.centroid_translation:
                save_to_pickle(data=self.translation_vector,
                               relative_path="../results/" + str(time.strftime('%Y-%m-%d')) + "/" + self.folder,
                               filename="training_data_centroid.pkl")
        else:
            raise ValueError("Train the model first.")

    def load_classifier(self, relative_path, folder=None, filename=None):
        """
        Loads a pre-trained classifier and sets the projector with the training seed.
        :param relative_path: relative path of the folder containing the list of trained classifiers.
                              It can be either TRAINED_MODELS or RESULTS
        :param filename: filename
        :return: list of trained classifiers
        """
        start_time = time.time()
        self.trained = True

        classifiers = []
        for i in self.random_seeds:
            proj_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes, test=self.test,
                                              data_format=self.data_format, dataset_name=self.dataset_name)
            classifiers.append(proj_classifier.load_classifier(relative_path=relative_path, folder=self.folder,
                                                               filename=self._set_baseline_filename(seed=i)))
        if self.centroid_translation:
            self.translation_vector = load_from_pickle(path=relative_path + self.folder + "training_data_centroid.pkl",
                                                       test=False)
        else:
            self.translation_vector = None

        print("\nLoading time: --- %s seconds ---" % (time.time() - start_time))

        self.classifiers = classifiers
        return classifiers


########
# MAIN #
########


def main(dataset_name, test, n_proj, size_proj, projection_mode, attack, eps, device):
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
                           data_format=data_format, dataset_name=dataset_name, test=test, epochs=None,
                           centroid_translation=False, library="cleverhans")
    # === train === #
    # model.train(x_train, y_train, device=device)
    # model.save_classifier(relative_path=RESULTS)

    # === load classifier === #
    # model.load_classifier(relative_path=TRAINED_MODELS)
    model.load_classifier(relative_path=RESULTS)

    # === evaluate === #
    seed=0
    baseline = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                               dataset_name=dataset_name, test=test, epochs=None, library="cleverhans")
    baseline.load_classifier(relative_path=RESULTS, filename=baseline.filename+"_seed="+str(seed))

    model.evaluate(x=x_test, y=y_test)
    for attack in ["fgsm", "pgd", "deepfool"]:#, "carlini", "deepfool", "newtonfool"]:
        x_test_adv = baseline.generate_adversaries(x=x_test, y=y_test, attack=attack, seed=seed)
        baseline.save_adversaries(data=x_test_adv, attack=attack, seed=seed)
        # x_test_adv = model.load_adversaries(relative_path=RESULTS, attack=attack)
        model.evaluate(x=x_test_adv, y=y_test)
        softmax_difference(classifier=model, x1=x_test, x2=x_test_adv)

    # x_test_adv = model.load_adversaries(dataset_name=dataset_name,attack=attack, eps=eps, test=test)
    # print("Distance from perturbations: ", compute_distances(x_test, x_test_adv, ord=model._get_norm(attack)))
    # model.evaluate(classifier=classifier, x=x_test_adv, y=y_test)

    # === generate perturbations === #
    # compute_variances(x_test, y_test)
    # projections, inverse_projections = model.compute_projections(input_data=x_test)
    #
    # projections, inverse_projections = model.compute_projections(input_data=x_test)
    # print(projections[0,0,0,0], inverse_projections[0,0,0,0])
    # # exit()
    # perturbations, augmented_inputs = compute_perturbations(input_data=x_test, inverse_projections=inverse_projections)

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
    # plot_images(image_data_list=[x_test,perturbations,augmented_inputs])

    # adversaries=[]
    # for method in ['fgsm', 'pgd', 'deepfool', 'carlini']:
    #     adversaries.append(model.load_adversaries(attack=method, eps=0.5))
    # plot_images(image_data_list=adversaries)

if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        n_proj_list = list(map(int, sys.argv[3].strip('[]').split(',')))
        size_proj_list = list(map(int, sys.argv[4].strip('[]').split(',')))
        projection_mode = sys.argv[5]
        attack = sys.argv[6]
        eps = float(sys.argv[7])
        device = sys.argv[8]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        n_proj_list = list(map(int, input("\nChoose the number of projections (type=list): ").strip('[]').split(',')))
        size_proj_list = list(map(int, input("\nChoose the size of projections (type=list): ").strip('[]').split(',')))
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")
        attack = input("\nChoose an attack ("+ATTACKS+"): ")
        eps = float(input("\nSet a ths for perturbation norm: "))
        device = input("\nChoose a device (cpu/gpu): ")

    for n_proj in n_proj_list:
        for size_proj in size_proj_list:
            K.clear_session()
            main(dataset_name=dataset_name, test=test, n_proj=n_proj, size_proj=size_proj,
                 projection_mode=projection_mode, attack=attack, eps=eps, device=device)

