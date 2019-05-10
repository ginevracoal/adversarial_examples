import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from utils import *
from classifier import AdversarialClassifier

SAVE_MODEL = False
MODEL_NAME = "baseline_convnet"
TRAINED_MODEL = "IBM-art/mnist_cnn_original.h5"

BATCH_SIZE = 128
EPOCHS = 12


class BaselineConvnet(AdversarialClassifier):
    """
    Simple convnet model. This will be our benchmark on the MNIST dataset.
    """

    def _set_layers(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, kernel_size=(3, 3),
                   activation='relu', data_format='channels_last')(inputs)
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

        model.summary()
        return model


def main():

    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_mnist()

    convNet = BaselineConvnet(input_shape=input_shape, num_classes=num_classes)

    classifier = convNet.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    #classifier = convNet.load_classifier(TRAINED_MODEL)

    convNet.evaluate_test(classifier, x_test, y_test)
    convNet.evaluate_adversaries(classifier, x_test, y_test)

    if SAVE_MODEL is True:
        convNet.save_model(classifier=classifier, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
