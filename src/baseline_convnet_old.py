# -*- coding: utf-8 -*-

"""
Simple CNN model. This is our benchmark on the MNIST dataset.
"""

from adversarial_classifier import *
from adversarial_classifier import AdversarialClassifier
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
import sys

############
# defaults #
############

MODEL_NAME = "baseline"


class BaselineConvnet(AdversarialClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, test):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        self.test = test
        self.dataset_name = dataset_name
        # self.sess = self._set_session()
        super(BaselineConvnet, self).__init__(input_shape, num_classes, data_format, dataset_name, test)

    def _set_session(self):
        K.clear_session()
        sess = tf.Session(graph=tf.Graph())
        sess.run(tf.global_variables_initializer())
        return sess

    def _set_model(self):

        if self.dataset_name == "mnist":

            inputs = Input(shape=self.input_shape)
            x = Conv2D(32, kernel_size=(3, 3),
                       activation='relu', data_format=self.data_format)(inputs)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(self.num_classes, activation='softmax')(x)
            model = Model(inputs=inputs, outputs=predictions)
            # model.compile(loss=keras.losses.categorical_crossentropy,
            #               optimizer=keras.optimizers.Adadelta(),
            #               metrics=['accuracy'])
            # model.summary()
            return model

        elif self.dataset_name == "cifar":

            # model = Sequential()
            # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
            #                  input_shape=self.input_shape))
            # model.add(BatchNormalization())
            # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            # model.add(BatchNormalization())
            # #model.add(MaxPooling2D((2, 2)))
            # model.add(Dropout(0.2))
            # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            # model.add(BatchNormalization())
            # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            # model.add(BatchNormalization())
            # model.add(MaxPooling2D((2, 2)))
            # model.add(Dropout(0.3))
            # model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            # model.add(BatchNormalization())
            # model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            # model.add(BatchNormalization())
            # model.add(MaxPooling2D((2, 2)))
            # model.add(Dropout(0.4))
            # model.add(Flatten())
            # model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
            # model.add(BatchNormalization())
            # model.add(Dropout(0.5))
            # predictions = model.add(Dense(10, activation='softmax'))
            # model = Model(inputs=inputs, outputs=predictions)

            # compile model
            # opt = SGD(lr=0.001, momentum=0.9)
            # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            inputs = Input(shape=self.input_shape)
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
            predictions = Dense(10, activation='softmax')(x)
            model = Model(inputs=inputs, outputs=predictions)

            return model

        def _get_logits(inputs):
            # todo: deprecated and only works for mnist
            x = Conv2D(32, kernel_size=(3, 3),
                       activation='relu', data_format=self.data_format)(inputs)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(self.num_classes, activation='softmax')(x)
            return predictions

    def save_classifier(self, classifier, model_name=MODEL_NAME):
        return super(BaselineConvnet, self).save_classifier(classifier=classifier,
                                                            model_name=self.dataset_name+"_"+model_name)

    def load_classifier(self, relative_path):
        """
        Loads a pretrained baseline classifier. It load either the baseline model or the adversarially trained robust version.
        :param dataset_name: dataset name
        returns: trained classifier
        """
        path = relative_path + str(self.dataset_name) + "_baseline.h5"
        return super(BaselineConvnet, self).load_classifier(path)

    def load_robust_classifier(self, relative_path, attack, eps):
        """
        Loads an adversarially trained robust classifier.
        :param dataset_name: dataset name
        :param attack: attack method for loading adversarially trained robust models
        :param eps: threshold for the norm of a perturbation
        returns: trained classifier
        """
        if attack == "deepfool":
            path = relative_path + "baseline/" + str(self.dataset_name) + "_" + str(attack) + "_robust_baseline.h5"
        else:
            if eps is None:
                raise ValueError("\nProvide a ths distance for the attacks.")
            path = relative_path + "baseline/" + str(self.dataset_name) + "_" + str(attack) + "_" + str(eps) \
                       + "_robust_baseline.h5"
        return super(BaselineConvnet, self).load_classifier(path)


def main(dataset_name, test, attack, eps):
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
    # classifier = model.train(x_train, y_train, batch_size=model.batch_size, epochs=model.epochs)
    # model.save_classifier(classifier = classifier, model_name = dataset_name+"_baseline")

    # === adversarial training === #
    # robust_classifier = model.adversarial_train(classifier, x_train, y_train, test=test, method=attack,
    #                                             batch_size=model.batch_size, epochs=model.epochs, dataset_name=dataset_name)
    # model.save_classifier(classifier=robust_classifier,
    #                       model_name=dataset_name+"_"+attack+"_"+str(model.eps)+"_robust_baseline")

    # === load classifier === #
    classifier = model.load_classifier(relative_path=TRAINED_MODELS+MODEL_NAME+"/")
    # robust_classifier = model.load_classifier(dataset_name=dataset_name, attack=attack, eps=eps)

    # === evaluations === #
    model.evaluate(classifier, x_test, y_test)

    x_test_adv = model.load_adversaries(attack=attack, dataset_name=dataset_name, eps=0.5, test=test)
    model.evaluate(classifier, x_test_adv, y_test)

    # model.evaluate(robust_classifier, x_test, y_test)
    #
    # x_test_adv = model.generate_adversaries(classifier=classifier, x=x_test, y=y_test, test=test, method=attack,
    #                                         dataset_name=dataset_name, eps=eps)
    # plot_images([x_test,x_test_adv])

    # model.save_adversaries(data=x_test_adv, dataset_name=dataset_name, attack=attack)
    # model.evaluate(classifier=classifier, x=x_test_adv, y=y_test)

    x_test_adv = model.load_adversaries(dataset_name=dataset_name,attack=attack,eps=eps,test=test)
    print("Distance from perturbations: ", compute_distances(x_test, x_test_adv, ord=model._get_norm(attack)))
    # plot_images([x_test,x_test_adv])#,np.array(x_test_adv,dtype=int)])

    for method in ['fgsm', 'pgd', 'deepfool','carlini']:
        x_test_adv = model.load_adversaries(attack=method, dataset_name=dataset_name, eps=0.5, test=test)
        model.evaluate(classifier, x_test_adv, y_test)
        # model.evaluate(robust_classifier, x_test_adv, y_test)


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        attack = sys.argv[3]
        eps = float(sys.argv[4])

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        attack = input("\nChoose an attack ("+ATTACKS+"): ")
        eps = float(input("\nSet a ths for perturbation norm: "))

    main(dataset_name=dataset_name, test=test, attack=attack, eps=eps)
    K.clear_session()

