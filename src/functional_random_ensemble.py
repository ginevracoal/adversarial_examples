"""
Buggy random ensemble model using Keras functional API. Currently using random_ensemble.py instead.
"""

# TODO: adjust or delete this

from art.classifiers import KerasClassifier
from baseline_convnet import BaselineConvnet
from keras.layers import Average
from keras.models import Model, Input
from utils import *


SAVE_MODEL = True
MODEL_NAME = "random_ensemble"
TRAINED_BASELINE = "IBM-art/mnist_cnn_original.h5"
TRAINED_MODELS = "../trained_models/"

BATCH_SIZE = 128
EPOCHS = 12
N_PROJECTIONS = 10
SIZE_PROJECTION = 8
SEED = 123

MIN = 0
MAX = 255


class RandomEnsemble(BaselineConvnet):
    """
    Classifies `n_proj` random projections of the training data in a lower dimensional space (whose dimension is
    `size_proj`^2), then classifies the original high dimensional data with a voting technique.
    """
    def __init__(self, input_shape, num_classes, n_proj, size_proj):
        """
        Extends BaselineConvnet initializer with additional informations about the projections.
        :param input_shape: full dimension input data shape
        :param num_classes: number of classes
        :param n_proj: number of random projections
        :param size_proj: size of a random projection
        """

        if size_proj > input_shape[1]:
            raise ValueError("The number of projections has to be lower than the image size.")

        super(RandomEnsemble, self).__init__(input_shape, num_classes)
        self.input_shape = (size_proj, size_proj, 1)
        self.n_proj = n_proj
        self.size_proj = size_proj
        self.projector = None
        self.trained = False

    def set_layers(self):
        """
        Using functional API
        :return:
        """

        # n_proj arrays of shape (dim_proj, dim_proj, 1)
        inputs = [Input(shape=self.input_shape) for i in range(self.n_proj)]

        baseline = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)

        # same model for all the projections
        outputs = [baseline.model(inputs[i]) for i in range(self.n_proj)]

        # final prediction as an average of the outputs
        prediction = Average()(outputs)

        model = Model(inputs=inputs, outputs=prediction, name=MODEL_NAME)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

        model.summary()

        return model

    def train(self, x_train, y_train, batch_size, epochs):
        """
        Trains the model defined by _set_layers over random projections of the training data

        :param x_train: original training data
        :param y_train: training labels
        :param batch_size:
        :param epochs:
        :param n_proj: number of projections
        :param dim_proj: dimension of a projection
        :return: trained classifier
        """

        x_train_projected = compute_projections(x_train, projector=self.projector,
                                                n_proj=self.n_proj, size_proj=self.size_proj)

        # TODO: indexing problem here
        classifier = KerasClassifier((MIN, MAX), model=self.model, use_logits=True)
        classifier.fit(x_train_projected, y_train, batch_size=batch_size, nb_epochs=epochs)

        self.trained = True

        return classifier


def main():

    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_mnist()

    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes,
                           n_proj=N_PROJECTIONS, size_proj=SIZE_PROJECTION)

    #classifier = model.load_classifier(relative_path=MODEL_NAME+"/")
    classifier = model.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    model.evaluate_test(classifier, x_test, y_test)
    model.evaluate_adversaries(classifier, x_test, y_test)

    if SAVE_MODEL is True:
        model.save_model(classifier=classifier, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
