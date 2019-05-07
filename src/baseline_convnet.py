import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from utils import *
from classifier import Classifier


SAVE_MODEL = False
MODEL_NAME = "baseline_convnet"
TRAINED_MODEL = "IBM-art/mnist_cnn_original.h5"

BATCH_SIZE = 128
EPOCHS = 12


class BaselineConvnet(Classifier):
    """
    Simple convnet model. This will be our benchmark on the MNIST dataset.
    """

    def _set_layers(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model


def main():

    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_mnist()

    # take a subset
    print("\nTaking just a small subset")
    x_train, y_train, x_test, y_test = x_train[:100], y_train[:100], x_test[:100], y_test[:100]

    convNet = BaselineConvnet(input_shape=input_shape, num_classes=num_classes)

    #classifier = convNet.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    classifier = convNet.load_classifier(TRAINED_MODEL)

    convNet.evaluate_test(classifier, x_test, y_test)
    convNet.evaluate_adversaries(classifier, x_test, y_test)

    if SAVE_MODEL is True:
        convNet.save_model(classifier=classifier, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
