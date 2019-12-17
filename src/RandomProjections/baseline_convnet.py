# -*- coding: utf-8 -*-

"""
Simple CNN model.
"""

import sys
sys.path.append(".")
from directories import *

from RandomProjections.adversarial_classifier import *
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from utils import _set_session
import random
from robustness_measures import softmax_difference, softmax_robustness
import copy

############
# defaults #
############

MODEL_NAME = "baseline"


class BaselineConvnet(AdversarialClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, test, library="art", epochs=None):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        super(BaselineConvnet, self).__init__(input_shape, num_classes, data_format, dataset_name, test, library, epochs)

    def _set_model_path(self, model_name="baseline"):
        """ Defines model path and filename """
        folder = str(model_name)+"/"
        if self.epochs == None:
            filename = self.dataset_name + "_" + str(model_name)
        else:
            filename = self.dataset_name + "_" + str(model_name) + "_epochs=" + str(self.epochs)
        return {'folder': folder, 'filename': filename}

    @staticmethod
    def _set_training_params(test, epochs):
        """
        Defines training parameters
        :param test: if True only takes a few samples
        :return: batch_size, epochs
        """
        batch_size = 10 if test else 100
        return {'batch_size': batch_size, 'epochs': epochs}

    def _get_logits(self, inputs):
        """
        Builds model architecture and returns logits layer
        :param inputs: input data
        :return: logits
        """
        inputs = tf.cast(inputs, tf.float32)
        if self.dataset_name == "mnist":
            x = Conv2D(32, kernel_size=(3, 3), activation='relu', data_format=self.data_format)(inputs)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            logits = Dense(self.num_classes, activation='softmax')(x)
            return logits

        elif self.dataset_name == "cifar":
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
            x = BatchNormalization()(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.2)(x)
            x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.3)(x)
            x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.4)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            logits = Dense(10, activation='softmax')(x)
            return logits

    def adversarial_train(self, x_train, y_train, device, attack, seed=0):
        """
        Performs adversarial training on the given classifier using an attack method. Training set adversaries are
        generated at training time on the baseline model.
        :param x_train: training data
        :param y_train: training labels
        :param attack: adversarial attack
        :param seed: seed used in baseline model training
        :return: adversarially trained classifier
        """
        random.seed(seed)
        if self.trained:
            robust_classifier = copy.deepcopy(self) #.load_classifier(relative_path=TRAINED_MODELS)
        else:
            robust_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                                test=self.test,
                                                data_format=self.data_format, dataset_name=self.dataset_name)
            robust_classifier.train(x_train, y_train, device)

        start_time = time.time()
        print("\n===== Adversarial training =====")
        x_train_adv = self.generate_adversaries(x_train, y_train, attack, seed=seed, device=device)
        # x_train_adv = self.load_adversaries(attack=attack, relative_path=DATA_PATH)

        # Data augmentation: expand the training set with the adversarial samples
        # x_train_ext = np.append(x_train, x_train_adv, axis=0)
        # y_train_ext = np.append(y_train, y_train, axis=0)

        # Retrain the CNN on the extended dataset
        # robust_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes, test=self.test,
        #                                     data_format=self.data_format, dataset_name=self.dataset_name)
        robust_classifier.train(x_train_adv, y_train, device)
        robust_classifier.filename = self._robust_classifier_name(attack=attack, eps=eps, seed=seed)

        print("Adversarial training time: --- %s seconds ---" % (time.time() - start_time))
        return robust_classifier

    def _robust_classifier_name(self, attack, seed, eps=None):
        """ Defines adversarially trained baseline filename """
        if eps:
            filename = self.filename + "_" + str(attack) + "_" + str(eps) + "_robust" +"_"+str(self.library)
        else:
            filename = self.filename + "_" + str(attack) + "_robust" +"_"+str(self.library)
        return filename

    def load_robust_classifier(self, relative_path, attack, seed=0, eps=None):
        """
        Loads an adversarially trained robust classifier.
        :param relative_path: relative path
        :param attack: attack method for loading adversarially trained robust models
        :param eps: threshold for the norm of a perturbation. If None,the default value from the baseclass is taken.
        returns: robust classifier
        """
        robust_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes, test=self.test,
                                            data_format=self.data_format, dataset_name=self.dataset_name)
        robust_classifier.filename = self._robust_classifier_name(attack=attack, eps=eps, seed=seed)
        return robust_classifier.load_classifier(relative_path=relative_path)

    def train_const_SGD(self, x_train, y_train, device, epochs, lr):
        """
        Perform SGD optimization with constant learning rate on a pre-trained network.
        :param x_train: training data
        :param y_train: training labels
        :param epochs: number of epochs
        :param lr: learning rate
        :return: re-trained network
        """
        if self.trained:
            self.filename = "SGD_lr=" + str(lr) + "_ep=" + str(epochs) + "_"+ self.dataset_name + "_baseline.h5"
            self.epochs = epochs
            self.model.compile(loss=keras.losses.categorical_crossentropy,
                                optimizer=keras.optimizers.SGD(lr=lr, clipnorm=1.),
                                metrics=['accuracy'])
            self.train(x_train=x_train, y_train=y_train, device=device)
            return self
        else:
            raise AttributeError("Train your classifier first.")

def plot_attacks(dataset_name, test, attacks, library):
    _set_session(device=device, n_jobs=1)
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                           test=test)
    model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                            dataset_name=dataset_name, test=test, library=library)
    model.load_classifier(relative_path=RESULTS, filename=model.filename+"_seed="+str(seed))
    images = []
    labels = []
    for attack in attacks:
        eps = model._get_attack_eps(dataset_name=model.dataset_name, attack=attack)
        x_test_adv = model.generate_adversaries(x=x_test, y=y_test, attack=attack, eps=eps)
        images.append(x_test_adv)
        avg_dist = compute_distances(x_test, x_test_adv, ord=model._get_norm(attack))['mean']
        labels.append(str(attack) + " avg_dist=" + str(avg_dist))

    plot_images(image_data_list=images,labels=labels)


def train_generate_eval(dataset_name, test, attacks, seed, device, library):
    random.seed(seed)
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                           test=test)

    print("\n === Baseline training === ")
    baseline = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                               dataset_name=dataset_name, test=test, epochs=None, library=library)
    baseline.load_classifier(relative_path=TRAINED_MODELS, filename=baseline.filename+"_seed="+str(seed))
    baseline.evaluate(x=x_test, y=y_test)

    print("\n === Adversarial training === ")
    robust_baselines = []
    for attack in attacks:
        print("\n", attack, "robust baseline")
        robust_baseline = baseline.adversarial_train(x_train, y_train, device, attack=attack, seed=seed)
        robust_baseline.save_classifier(relative_path=RESULTS)
        # robust_baseline = baseline.load_robust_classifier(relative_path=TRAINED_MODELS, attack=attack, seed=seed)
        robust_baseline.evaluate(x=x_test, y=y_test)
        robust_baselines.append(robust_baseline)

    print("\n === Adversarial evaluations === ")
    for attack in attacks:
        x_test_adv = baseline.generate_adversaries(x=x_test, y=y_test, attack=attack, seed=seed)
        baseline.save_adversaries(data=x_test_adv, attack=attack, seed=seed)
        baseline.evaluate(x=x_test_adv, y=y_test)
        softmax_difference(classifier=baseline, x1=x_test, x2=x_test_adv)
        for idx, attack in enumerate(attacks):
            print("\nAttack against", attack, "robust baseline")
            robust_baselines[idx].evaluate(x=x_test_adv, y=y_test)


def train_eval(dataset_name, test, seed, device, attacks, library):
    random.seed(seed)
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                           test=test)
    baseline = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                               dataset_name=dataset_name, test=test, epochs=None, library=library)
    baseline.train(x_train, y_train, device)
    baseline.save_classifier(relative_path=RESULTS, filename=baseline.filename+"_seed="+str(seed))
    # baseline.load_classifier(relative_path=TRAINED_MODELS)
    baseline.evaluate(x=x_test, y=y_test)
    for attack in attacks:
        x_test_adv = baseline.generate_adversaries(x=x_test, y=y_test, attack=attack, seed=seed)
        baseline.save_adversaries(data=x_test_adv, attack=attack, seed=seed)
        # x_test_adv = baseline.load_adversaries(attack=attack, relative_path=DATA_PATH, seed=0)
        softmax_difference(classifier=baseline, x1=x_test, x2=x_test_adv)
        softmax_robustness(classifier=baseline, x1=x_test, x2=x_test_adv)
        baseline.evaluate(x=x_test_adv, y=y_test)


def robust_eval(dataset_name, test, seed, device, robust_models, attacks, library):
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                           test=test)
    baseline = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                               dataset_name=dataset_name, test=test, epochs=None, library=library)
    baseline.load_classifier(relative_path=TRAINED_MODELS)
    baseline.evaluate(x=x_test, y=y_test)
    for method in robust_models:
        print("\n===",method,"robust eval ===")
        robust_baseline = baseline.load_robust_classifier(relative_path=TRAINED_MODELS, attack=method, seed=seed)
        robust_baseline.evaluate(x=x_test, y=y_test)
        for attack in attacks:
            for seed in [0,1,2]:
                print("\nseed =", seed)
                x_test_adv = baseline.load_adversaries(attack=attack, relative_path=DATA_PATH, seed=seed)
                robust_baseline.evaluate(x=x_test_adv, y=y_test)


def main(dataset_name, test, device, seed):
    """
    :param dataset: choose between "mnist" and "cifar"
    :param test: if True, only takes a few input samples.
    :param device: training device (cpu/gpu)
    """
    # === GPU keras session === #
    _set_session(device=device, n_jobs=2)

    # robust_models = [""]
    attacks = ["fgsm","pgd","deepfool","virtual","spatial","saliency"]
    # plot_attacks(dataset_name=dataset_name, test=test, attacks=attacks, library="cleverhans")
    # train_eval_attacks(dataset_name=dataset_name, test=test, attacks = attacks, seed=seed, device=device,
    #                    library="cleverhans")
    train_generate_eval(dataset_name=dataset_name, test=test, seed=seed, device=device, attacks=attacks,
                        library="cleverhans")
    # robust_eval(dataset_name=dataset_name, test=test, seed=seed, device=device, robust_models=attacks, attacks=attacks,
    #             library="cleverhans")

    # === initialize === #
    # x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
    #                                                                                        test=test)
    # model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
    #                         dataset_name=dataset_name, test=test)

    # === training === #
    # model.train(x_train, y_train, device)
    # model.train_const_SGD(x_train=x_train, y_train=y_train, device=device, epochs=50, lr=0.01)
    # model.save_classifier(relative_path=RESULTS)

    # === load classifier === #
    # model.load_classifier(relative_path=RESULTS)
    # model.load_classifier(relative_path=TRAINED_MODELS)
    # model.load_robust_classifier(relative_path=TRAINED_MODELS, attack=attack)

    # === adversarial training === #
    # model = model.adversarial_train(x_train, y_train, device=device, attack=attack)
    # model.save_classifier(relative_path=RESULTS)

    # === evaluate  === #
    # model.evaluate(x=x_test, y=y_test)
    # for attack in attacks:
    #     x_test_adv = model.generate_adversaries(x=x_test, y=y_test, attack=attack, seed=seed, eps=eps)
    #     model.save_adversaries(data=x_test_adv, attack=attack, seed=seed, eps=eps)
    #     x_test_adv = model.load_adversaries(attack=attack, relative_path=DATA_PATH, seed=seed)
    #     model.evaluate(x=x_test_adv, y=y_test)


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        device = sys.argv[3]
        seed = int(sys.argv[4])

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        device = input("\nChoose a device (cpu/gpu): ")
        seed = input("\nSet a training seed (type=int): ")

    main(dataset_name=dataset_name, test=test, device=device, seed=seed)

