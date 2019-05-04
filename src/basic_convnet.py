import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod


SAVE_MODEL = False
MODEL_NAME = "basic_convnet"
TRAINED_MODELS = "../trained_models/"

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 12
IMG_COLS = 28
IMG_ROWS = 28
MIN = 0
MAX = 255


def preprocess_mnist():
    """Preprocess mnist for keras training"""

    # input image dimensions
    img_rows, img_cols = IMG_ROWS, IMG_COLS

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test, input_shape


def basic_convnet(input_shape):
    """
    Simple convnet model. This will be our benchmark on the MNIST dataset.
    """

    batch_size = BATCH_SIZE
    num_classes = NUM_CLASSES
    epochs = EPOCHS

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model, batch_size, epochs


def train_model(x_train, y_train, x_test, y_test, input_shape):
    model, batch_size, epochs = basic_convnet(input_shape)

    classifier = KerasClassifier((MIN, MAX), model=model)
    classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs)

    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy: %.2f%%" % (acc * 100))

    return classifier


def load_trained_model():
    # load a trained model
    classifier_model = load_model("../trained_models/IBM-art/mnist_cnn_original.h5")
    classifier = KerasClassifier((MIN, MAX), classifier_model, use_logits=False)
    return classifier


def main():

    x_train, y_train, x_test, y_test, input_shape = preprocess_mnist()

    # subset
    x_train, y_train, x_test, y_test = x_train[:100], y_train[:100], x_test[:100], y_test[:100]

    # classifier = train_model(x_train, y_train, x_test, y_test, input_shape)
    classifier = load_trained_model()

    # predict the first 100 images
    x_test_pred = np.argmax(classifier.predict(x_test), axis=1)
    correct_preds = np.sum(x_test_pred == np.argmax(y_test, axis=1))

    print("\nOriginal test data (first 100 images):")
    print("Correctly classified: {}".format(correct_preds))
    print("Incorrectly classified: {}".format(len(x_test) - correct_preds))

    # generate adversarial examples using FGSM
    attacker = FastGradientMethod(classifier, eps=0.5)
    x_test_adv = attacker.generate(x_test)

    # evaluate the performance
    x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
    nb_correct_adv_pred = np.sum(x_test_adv_pred == np.argmax(y_test, axis=1))

    print("\nAdversarial test data (first 100 images):")
    print("Correctly classified: {}".format(nb_correct_adv_pred))
    print("Incorrectly classified: {}".format(len(x_test) - nb_correct_adv_pred))

    if SAVE_MODEL is True:
        classifier.save(filename=MODEL_NAME, path=TRAINED_MODELS)


if __name__ == "__main__":

    main()
