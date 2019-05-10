import keras
from keras.callbacks import History
from keras.engine import training
from keras.layers import Average, Concatenate
#from art.classifiers import KerasClassifier
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Model, Input
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List
from utils import *
from classifier import AdversarialClassifier
from baseline_convnet import BaselineConvnet

SAVE_MODEL = False
MODEL_NAME = "random_ensemble"
# TRAINED_MODEL_PATH =

BATCH_SIZE = 128
EPOCHS = 12
N_PROJECTIONS = 10
SIZE_PROJECTION = 8


class RandomEnsemble(AdversarialClassifier):
    """
    Classifies n_proj random projections of the training data in a lower dimensional space (whose dimension is
    dim_proj=size_proj^2), then classifies the original high dimensional data with a voting technique.
    """
    def __init__(self, input_shape, num_classes, n_proj, size_proj, *args, **kwargs):
        self.n_proj = n_proj
        self.size_proj = size_proj
        self.num_classes = num_classes
        #super(RandomEnsemble, self).__init__(input_shape, num_classes, *args, **kwargs)
        self.input_shape = (size_proj, size_proj, 1)
        self.model = self._set_layers()

    def _set_layers(self):
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

        model = Model(inputs=inputs, outputs=prediction, name='random_ensemble')
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

        model.summary()

        return model

    def train(self, x_train, y_train, batch_size, epochs):
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

        x_train_projected = compute_random_projections(x_train, n_proj=self.n_proj, size_proj=self.size_proj)

        # TODO: indexing problem here
        classifier = KerasClassifier((MIN, MAX), model=self.model, use_logits=True)
        classifier = AdversarialClassifier(model=self.model, use_logits=True)
        classifier.fit(x_train_projected, y_train, batch_size=batch_size, nb_epochs=epochs)

        return classifier


def main():

    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_mnist()

    # take a subset
    print("\nTaking just the first 100 images.")
    x_train, y_train, x_test, y_test = x_train[:100], y_train[:100], x_test[:100], y_test[:100]

    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes,
                           n_proj=N_PROJECTIONS, size_proj=SIZE_PROJECTION)

    classifier = model.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    #classifier = model.load_classifier(TRAINED_MODEL)

    exit()
    model.evaluate_test(classifier, x_test, y_test)
    model.evaluate_adversaries(classifier, x_test, y_test)

    if SAVE_MODEL is True:
        model.save_model(classifier=classifier, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
