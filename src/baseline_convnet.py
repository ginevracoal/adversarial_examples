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
MODEL_NAME = "baseline_convnet"


class BaselineConvnet(AdversarialClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        self.dataset_name = dataset_name
        self.batch_size, self.epochs = self._set_training_params()
        super(BaselineConvnet, self).__init__(input_shape, num_classes, data_format)

    def _set_training_params(self):
        if self.dataset_name == "mnist":
            return 128, 12
        elif self.dataset_name == "cifar":
            return 128, 120

    def _set_layers(self):

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
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
            # model.summary()
            return model

        elif self.dataset_name == "cifar":

            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                             input_shape=self.input_shape))
            model.add(BatchNormalization())
            model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(BatchNormalization())
            model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.3))
            model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(BatchNormalization())
            model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.4))
            model.add(Flatten())
            model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='softmax'))
            # compile model
            opt = SGD(lr=0.001, momentum=0.9)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            return model

def main(dataset_name, test, attack):
    """
    :param dataset: choose between "mnist" and "cifar"
    :param test: if True, only takes the first 100 samples.
    :param attack: choose between "fgsm", "pgd", "deepfool", "carlini_linf", "virtual", "newtonfool"
    """

    # load dataset #
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name, test=test)

    model = BaselineConvnet(input_shape=input_shape, num_classes=num_classes, data_format=data_format, dataset_name=dataset_name)

    # train classifier #
    # classifier = model.train(x_train, y_train, batch_size=model.batch_size, epochs=model.epochs)
    # model.save_model(classifier = classifier, model_name = dataset_name+"_baseline")

    # load classifier #
    rel_path = TRAINED_MODELS+"baseline/"+str(dataset_name)+"_baseline.h5"
    # rel_path = RESULTS+time.strftime('%Y-%m-%d') + "/" + str(dataset_name)+"_baseline.h5"
    classifier = model.load_classifier(relative_path=rel_path)

    # rel_path = TRAINED_MODELS+"baseline/"+str(dataset_name)+"_"+str(attack)+"_robust_baseline.h5"
    # robust_classifier = model.load_classifier(relative_path=rel_path)

    # adversarial training #
    # robust_classifier = model.adversarial_train(classifier, x_train, y_train, test=test, method=attack,
    #                                             batch_size=model.batch_size, epochs=model.epochs, dataset_name=dataset_name)
    # model.save_model(classifier = robust_classifier, model_name = dataset_name+"_"+attack+"_robust_baseline")

    # evaluations #
    # model.evaluate_test(classifier, x_test, y_test)
    model.evaluate_test(robust_classifier, x_test, y_test)

    #############
    # todo: solve this bug eventually... not urgent
    # notice: here we are actually not only saving x_test, but also y_test... This is not a problem since pkl
    # loading deals with this issue. the correct code should be:
    # x_test_adv, _ = model.evaluate_adversaries(...)
    #############

    for method in ['fgsm', 'pgd', 'deepfool', 'carlini_linf']:
        # x_test_adv = model.evaluate_adversaries(classifier=classifier, x_test=x_test, y_test=y_test,
        x_test_adv = model.evaluate_adversaries(classifier=robust_classifier, x_test=x_test, y_test=y_test,
                                                method=method, test=test, dataset_name=dataset_name)
    # save_to_pickle(data=x_test_adv, filename=dataset_name+"_x_test_"+attack+".pkl")

    # for method in ['fgsm','pgd','deepfool','carlini_linf']:
    #     x_test_adv = model.evaluate_adversaries(classifier, x_test, y_test, method=method, dataset_name=dataset_name,
    #                                             adversaries_path=DATA_PATH+dataset_name+"_x_test_"+attack+".pkl", test=test)

if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        attack = sys.argv[3]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        attack = input("\nChoose an attack ("+ATTACKS+"): ")

    K.clear_session()
    main(dataset_name=dataset_name, test=test, attack=attack)
