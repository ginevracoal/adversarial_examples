# -*- coding: utf-8 -*-

"""
Simple CNN model. This is our benchmark on the MNIST dataset.
"""

import sys
from adversarial_classifier import *
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

############
# defaults #
############

MODEL_NAME = "baseline"
TRAINED_MODELS = "../trained_models/baseline/"


class BaselineConvnet(AdversarialClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, test):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        super(BaselineConvnet, self).__init__(input_shape, num_classes, data_format, dataset_name, test)
        self.model_name = self.dataset_name + "_baseline"

    @staticmethod
    def _set_training_params(test):
        """
        Defines training parameters
        :param test: if True only takes the first 100 samples
        :return: batch_size, epochs
        """
        if test:
            return {'batch_size': 100, 'epochs': 1}
        else:
            return {'batch_size': 500, 'epochs': 100}

    def _get_logits(self, inputs):
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

    def adversarial_train(self, x_train, y_train, device, attack, eps=0.5):
        """
        Performs adversarial training on the given classifier using an attack method. By default adversaries are
        generated at training time and epsilon is set to 0.5.
        :param attack: adversarial attack
        :return: adversarially trained classifier
        # todo: docstring
        """

        start_time = time.time()
        print("\n===== Adversarial training =====")
        # generate adversarial examples on train and test sets
        x_train_adv = self.generate_adversaries(x_train, y_train, attack)

        # Data augmentation: expand the training set with the adversarial samples
        x_train_ext = np.append(x_train, x_train_adv, axis=0)
        y_train_ext = np.append(y_train, y_train, axis=0)

        # Retrain the CNN on the extended dataset
        robust_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes, test=self.test,
                                            data_format=self.data_format, dataset_name=self.dataset_name)
        robust_classifier.train(x_train_ext, y_train_ext, device)
        print("\nAdversarial training time: --- %s seconds ---" % (time.time() - start_time))

        return robust_classifier

    def load_robust_classifier(self, relative_path, attack, eps=0.5):
        """
        Loads an adversarially trained robust classifier.
        :param relative_path: relative path
        :param attack: attack method for loading adversarially trained robust models
        :param eps: threshold for the norm of a perturbation
        returns: trained classifier
        """
        robust_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes, test=self.test,
                                            data_format=self.data_format, dataset_name=self.dataset_name)
        robust_classifier.model_name = str(attack) + "_robust_" + self.model_name

        # todo: update filenames as additional infos + self.model_name
        if attack == "deepfool":
            robust_classifier.model_name = str(attack) + "_robust_" + self.model_name
            return robust_classifier.load_classifier(relative_path=relative_path)
        else:
            if eps is None:
                raise ValueError("\nProvide a ths distance for the attacks.")
            else:
                # todo: add eps in robust models filenames
                robust_classifier.model_name = str(self.dataset_name) + "_" + str(attack) + "_robust_" + MODEL_NAME
                return robust_classifier.load_classifier(relative_path=relative_path)


def main(dataset_name, test, attack, eps, device):
    """
    :param dataset: choose between "mnist" and "cifar"
    :param test: if True, only takes the first 100 samples.
    :param attack: attack name
    :param eps: threshold for perturbation norm.
    """

    # === initialize === #
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name, test=test)
    model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                            dataset_name=dataset_name, test=test)

    # === training === #
    # model.train(x_train, y_train, device)
    # model.save_classifier(relative_path=RESULTS)

    # === load classifier === #
    # model.load_classifier(relative_path=RESULTS)
    # model.load_classifier(relative_path=TRAINED_MODELS)
    model.load_classifier(relative_path="../trained_models/baseline/")
    # robust_classifier = model.load_robust_classifier(relative_path=TRAINED_MODELS, attack=attack, eps=eps)

    # === adversarial training === #
    # robust_classifier = model.adversarial_train(x_train, y_train, device=device, attack="fgsm")
    # robust_classifier.save_classifier(relative_path=RESULTS)

    # === evaluations === #
    model.evaluate(x_test, y_test)
    # robust_classifier.evaluate(x_test, y_test)

    # x_test_adv = model.generate_adversaries(x=x_test, y=y_test, attack=attack, eps=eps)
    # model.save_adversaries(data=x_test_adv, attack=attack, eps=eps)
    # model.evaluate(x=x_test_adv, y=y_test)

    # x_test_adv = model.load_adversaries(attack=attack,eps=eps)
    # print("Distance from perturbations: ", compute_distances(x_test, x_test_adv, ord=model._get_norm(attack)))
    # plot_images([x_test,x_test_adv])#,np.array(x_test_adv,dtype=int)])

    for method in ['fgsm', 'pgd', 'deepfool','carlini']:
        x_test_adv = model.load_adversaries(attack=method, eps=0.5)
        model.evaluate(x_test_adv, y_test)
        # model.evaluate(robust_classifier, x_test_adv, y_test)


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        attack = sys.argv[3]
        eps = float(sys.argv[4])
        device = sys.argv[5]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        attack = input("\nChoose an attack ("+ATTACKS+"): ")
        eps = float(input("\nSet a ths for perturbation norm: "))
        device = input("\nChoose a device (cpu/gpu): ")

    main(dataset_name=dataset_name, test=test, attack=attack, eps=eps, device=device)

