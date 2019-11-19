# -*- coding: utf-8 -*-

"""
Simple CNN model. This is our benchmark on the MNIST dataset.
"""

import sys
from adversarial_classifier import *
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from utils import _set_session

############
# defaults #
############

MODEL_NAME = "baseline"


class BaselineConvnet(AdversarialClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, test, epochs=None):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        super(BaselineConvnet, self).__init__(input_shape, num_classes, data_format, dataset_name, test, epochs)

    def _set_model_path(self):
        folder = "baseline/"
        if self.epochs == None:
            filename = self.dataset_name + "_" + MODEL_NAME
        else:
            filename = self.dataset_name + "_" + MODEL_NAME + "_epochs=" + str(self.epochs)
        return {'folder': folder, 'filename': filename}

    @staticmethod
    def _set_training_params(test, epochs):
        """
        Defines training parameters
        :param test: if True only takes the first 100 samples
        :return: batch_size, epochs
        """
        batch_size = 100 #if test else 500
        return {'batch_size': batch_size, 'epochs': epochs}

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

    def adversarial_train(self, x_train, y_train, device, attack, eps=None):
        """
        Performs adversarial training on the given classifier using an attack method. By default adversaries are
        generated at training time and epsilon is set to 0.5.
        :param attack: adversarial attack
        :return: adversarially trained classifier
        # todo: docstring
        """

        start_time = time.time()
        print("\n===== Adversarial training =====")
        eps = self._get_attack_eps(dataset_name=self.dataset_name, attack=attack)
        self.trained = True
        x_train_adv = self.generate_adversaries(x_train, y_train, attack, eps)
        # x_train_adv = self.load_adversaries(attack=attack, eps=eps)

        # Data augmentation: expand the training set with the adversarial samples
        x_train_ext = np.append(x_train, x_train_adv, axis=0)
        y_train_ext = np.append(y_train, y_train, axis=0)

        # Retrain the CNN on the extended dataset
        robust_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes, test=self.test,
                                            data_format=self.data_format, dataset_name=self.dataset_name)
        robust_classifier.train(x_train_ext, y_train_ext, device)
        robust_classifier.filename = self._robust_classifier_name(attack=attack, eps=eps)

        print("Adversarial training time: --- %s seconds ---" % (time.time() - start_time))
        return robust_classifier

    def _robust_classifier_name(self, attack, eps=None):
        if eps:
            filename = self.filename + "_" + str(attack) + "_" + str(eps) + "_robust"
        else:
            eps = self._get_attack_eps(dataset_name=self.dataset_name, attack=attack)
            if eps:
                filename = self.filename + "_" + str(attack) + "_" + str(eps) + "_robust"
            else:
                filename = self.filename + "_" + str(attack) + "_robust"
        return filename

    def load_robust_classifier(self, relative_path, attack, eps=None):
        """
        Loads an adversarially trained robust classifier.
        :param relative_path: relative path
        :param attack: attack method for loading adversarially trained robust models
        :param eps: threshold for the norm of a perturbation
        returns: trained classifier
        """
        robust_classifier = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes,
                                            test=self.test,
                                            data_format=self.data_format, dataset_name=self.dataset_name)
        self.trained = True
        return robust_classifier.load_classifier(relative_path=relative_path,
                                                 filename=self._robust_classifier_name(attack=attack, eps=eps))

    def train_const_SGD(self, x_train, y_train, device, epochs, lr):
        """
        Perform SGD optimization with constant learning rate on a pre-trained network.
        :param epochs: number of epochs
        :param lr: learning rate
        :return: re-trained network
        """
        if self.trained:
            self.filename = "SGD_lr=" + str(lr) + "_ep=" + str(epochs) + "_" + self.filename
            self.epochs = epochs
            self.model.compile(loss=keras.losses.categorical_crossentropy,
                                optimizer=keras.optimizers.SGD(lr=lr, clipnorm=1.),
                                metrics=['accuracy'])
            self.train(x_train=x_train, y_train=y_train, device=device)
            return self
        else:
            raise AttributeError("Train your classifier first.")

def main(dataset_name, test, attack, eps, device):
    """
    :param dataset: choose between "mnist" and "cifar"
    :param test: if True, only takes the first 100 samples.
    :param attack: attack name
    :param eps: threshold for perturbation norm.
    """

    # === initialize === #
    _set_session(device=device, n_jobs=1)
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                           test=test)
    model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                            dataset_name=dataset_name, test=test)

    # === training === #
    # model.train(x_train, y_train, device)
    # model.save_classifier(relative_path=RESULTS)
    # exit()
    # === load classifier === #
    # model.load_classifier(relative_path=RESULTS)
    model.load_classifier(relative_path=TRAINED_MODELS)
    model.train_const_SGD(x_train=x_train, y_train=y_train, device=device, epochs=50, lr=0.01)
    # model.load_robust_classifier(relative_path=TRAINED_MODELS, attack=attack, eps=eps)

    # === adversarial training === #
    # model = model.adversarial_train(x_train, y_train, device=device, attack=attack)
    model.save_classifier(relative_path=RESULTS)

    # === evaluations === #
    # model.load_robust_classifier(relative_path=RESULTS, attack=attack)
    model.evaluate(x=x_test, y=y_test)

    seed = 0
    # images = []
    # labels = []
    attacks = ["fgsm","pgd","carlini","deepfool","newtonfool"]
    for attack in attacks:
        # x_test_adv = model.generate_adversaries(x=x_test, y=y_test, attack=attack, seed=seed, eps=eps)
        # model.save_adversaries(data=x_test_adv, attack=attack, seed=seed, eps=eps)

        x_test_adv = model.load_adversaries(attack=attack, seed=seed,
                                            eps=model._get_attack_eps(dataset_name=dataset_name, attack=attack))
        model.evaluate(x=x_test_adv, y=y_test)

        # images.append(x_test_adv)
        # avg_dist = compute_distances(x_test, x_test_adv, ord=model._get_norm(attack))['mean']
        # labels.append(str(attack) + " avg_dist=" + str(avg_dist))

    # plot_images(image_data_list=images,labels=labels)


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

