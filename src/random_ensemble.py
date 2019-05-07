import keras
from keras.callbacks import History
from keras.engine import training
from keras.layers import Average, Concatenate
from art.classifiers import KerasClassifier
from keras.models import Model, Input
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List
from utils import *
from classifier import Classifier
from baseline_convnet import BaselineConvnet

SAVE_MODEL = False
MODEL_NAME = "random_ensemble"
# TRAINED_MODEL_PATH =

BATCH_SIZE = 128
EPOCHS = 12
N_PROJECTIONS = 10
DIM_PROJECTION = 8


class RandomEnsemble(Classifier):
    """
    Classifies n_proj random projections of the training data in a lower dimensional space,
    then classifies the original high dimensional data with a voting technique.
    """
    def __init__(self, input_shape, num_classes, n_proj, dim_proj):
        self.n_proj = n_proj
        self.num_classes = num_classes
        self.dim_proj = dim_proj
        self.input_shape = (n_proj, dim_proj, dim_proj, 1)
        self.model = self._set_layers()

    def _set_layers(self):
        """
        Using functional API
        :return:
        """

        #inputs = [Input(shape=self.input_shape) for i in range(self.n_proj)]

        inputs = Input(shape=self.input_shape)

        # TODO: n_proj reti diverse, una su ogni proiezione
        # same model for all the projections
        baseline_model = BaselineConvnet(input_shape=self.input_shape[1:], num_classes=self.num_classes).model

        # devo dire di prendere l'i-esimo input
        outputs = [baseline_model.outputs[0] for i in range(self.n_proj)]

        #outputs = [model.outputs[0] for model in models]
        # final prediction as an average of the outputs
        prediction = Average()(outputs)

        model = Model(inputs=inputs, outputs=prediction, name='ensemble')

        return model

    def ensemble_training(self, x_train, y_train, batch_size, epochs):
        """

        :param x_train: original training data
        :param y_train: training labels
        :param batch_size: 
        :param epochs: 
        :param n_proj: number of projections
        :param dim_proj: dimension of a projection
        :return: trained classifier
        """

        # TODO: docstring

        x_train_projected = compute_random_projections(x_train, n_proj=self.n_proj, dim_proj=self.dim_proj)

        classifier = KerasClassifier((MIN, MAX), model=self.model)
        classifier.fit(x_train_projected, y_train, batch_size=batch_size, nb_epochs=epochs)

        return classifier


def main():

    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_mnist()

    # take a subset
    print("\nTaking just the first 100 images.")
    x_train, y_train, x_test, y_test = x_train[:100], y_train[:100], x_test[:100], y_test[:100]

    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes,
                           n_proj=N_PROJECTIONS, dim_proj=DIM_PROJECTION)
    exit()

    classifier = model.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    #classifier = model.load_classifier(TRAINED_MODEL)

    model.evaluate_test(classifier, x_test, y_test)
    model.evaluate_adversaries(classifier, x_test, y_test)

    if SAVE_MODEL is True:
        model.save_model(classifier=classifier, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
