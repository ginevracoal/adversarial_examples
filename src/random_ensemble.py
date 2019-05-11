import keras
import numpy as np
from keras.layers import Average
from art.classifiers import KerasClassifier
from keras.models import Model, Input
from art.attacks import FastGradientMethod
from utils import *
from baseline_convnet import BaselineConvnet


SAVE_MODEL = False
MODEL_NAME = "random_ensemble"
TRAINED_BASELINE = "IBM-art/mnist_cnn_original.h5"


BATCH_SIZE = 128
EPOCHS = 12
N_PROJECTIONS = 10
SIZE_PROJECTION = 8


class RandomEnsemble(BaselineConvnet):
    """
    Classifies n_proj random projections of the training data in a lower dimensional space (whose dimension is
    dim_proj=size_proj^2), then classifies the original high dimensional data with a voting technique.
    """
    def __init__(self, input_shape, num_classes, n_proj, size_proj):
        """
        Extends BaselineConvnet initializer with informations about the projections.
        """
        super(BaselineConvnet, self).__init__(input_shape, num_classes)
        self.input_shape = (size_proj, size_proj, 1)
        self.n_proj = n_proj
        self.size_proj = size_proj
        self.projector = None
        self.classifier = None

    def buggy_set_layers(self):
        # TODO: delete this
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

    def buggy_train(self, x_train, y_train, batch_size, epochs):
        # TODO: delete this
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

        x_train_projected = compute_projections(x_train, n_proj=self.n_proj, size_proj=self.size_proj)

        # TODO: indexing problem here
        classifier = KerasClassifier((MIN, MAX), model=self.model, use_logits=True)
        classifier.fit(x_train_projected, y_train, batch_size=batch_size, nb_epochs=epochs)

        return classifier

    def _ensemble_classifier(self, classifiers, projected_data):
        predictions = np.array([classifier.predict(projected_data[i]) for i, classifier in enumerate(classifiers)])
        summed_predictions = np.sum(predictions, axis=0)
        return summed_predictions

    def train(self, x_train, y_train, batch_size, epochs):
        """
        Trains the baseline model over n_proj random projections of the training data

        :param x_train:
        :param y_train:
        :param batch_size:
        :param epochs:
        :return: list of n_proj trained models
        """

        self.projector = GaussianRandomProjection(n_components=self.size_proj * self.size_proj)
        x_train_projected = compute_projections(x_train, self.projector, n_proj=self.n_proj, size_proj=self.size_proj)

        # use the same model for all trainings
        convNet = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)

        # train n_proj classifiers on different training data
        classifier = [convNet.train(x_train_projected[i], y_train, batch_size=batch_size, epochs=epochs) for i in
                      range(len(x_train_projected))]

        self.classifier = classifier
        return classifier

    def train(self, x_train, y_train, batch_size, epochs):
        """
        Trains the baseline model over n_proj random projections of the training data

        :param x_train:
        :param y_train:
        :param batch_size:
        :param epochs:
        :return: list of n_proj trained models
        """

        self.projector = GaussianRandomProjection(n_components=self.size_proj * self.size_proj)
        x_train_projected = compute_projections(x_train, self.projector, n_proj=self.n_proj, size_proj=self.size_proj)

        # use the same model for all trainings
        convNet = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)

        # train n_proj classifiers on different training data
        classifier = [convNet.train(x_train_projected[i], y_train, batch_size=batch_size, epochs=epochs) for i in
                      range(len(x_train_projected))]

        return classifier

    def predict(self, classifier, x_test):
        """
        Compute the average prediction over the trained models.

        :param classifier: list of trained classifiers over different projections
        :param x_test: list of projected test data
        :return:
        """
        #x_test_projected = compute_random_projections(x_test, n_proj=self.n_proj, size_proj=self.size_proj)
        # TODO: raise error if there is no projector

        x_test_projected = compute_projections(x_test, self.projector, n_proj=self.n_proj, size_proj=self.size_proj)

        predictions = self._ensemble_classifier(classifier, x_test_projected)

        return predictions

    def _evaluate_test(self, classifier, x_test, y_test):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        """
        preds = self.predict(classifier, x_test)
        # preds = self._ensemble_classifier(classifier, x_test_projected)
        # TODO: refactor this
        # numeric test
        y_test = np.argmax(y_test, axis=1)

        for i in range(len(y_test)):
            print(preds[i], y_test[i])
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print("\nTest accuracy: %.2f%%" % (acc * 100))

    def evaluate_adversaries(self, classifier, x_test, y_test):
        """
        Evaluates the trained model against FGSM and prints the number of misclassifications.
        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        :return:
        x_test_pred: test set predictions
        x_test_adv: adversarial perturbations of test data
        x_test_adv_pred: adversarial test set predictions
        """
        # TODO: implementare tutto con questa sintassi, che è quella corretta
        x_test_pred = np.argmax(self.predict(classifier, x_test), axis=1)

        correct_preds = np.sum(x_test_pred == np.argmax(y_test, axis=1))

        print("\nOriginal test data:")
        print("Correctly classified: {}".format(correct_preds))
        print("Incorrectly classified: {}".format(len(x_test) - correct_preds))

        # generate adversarial examples using FGSM on the baseline model
        convNet = BaselineConvnet(input_shape=self.input_shape, num_classes=self.num_classes)
        baseline_classifier = convNet.load_classifier(TRAINED_BASELINE)
        attacker = FastGradientMethod(baseline_classifier, eps=0.5)
        x_test_adv = attacker.generate(x_test)

        # evaluate the performance
        x_test_adv_pred = np.argmax(self.predict(classifier, x_test_adv), axis=1)
        nb_correct_adv_pred = np.sum(x_test_adv_pred == np.argmax(y_test, axis=1))

        print("\nAdversarial test data:")
        print("Correctly classified: {}".format(nb_correct_adv_pred))
        print("Incorrectly classified: {}".format(len(x_test) - nb_correct_adv_pred))

        return x_test_pred, x_test_adv, x_test_adv_pred


def main():

    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_mnist()

    # take a subset
    print("\nTaking just the first 100 images.")
    x_train, y_train, x_test, y_test = x_train[:100], y_train[:100], x_test[:100], y_test[:100]

    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes,
                           n_proj=N_PROJECTIONS, size_proj=SIZE_PROJECTION)

    #classifier = model.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    #classifier = model.load_classifier(TRAINED_MODEL)
    classifier = model.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    model.evaluate_test(classifier, x_test, y_test)

    #TODO: evaluate adversaries
    model.evaluate_adversaries(classifier, x_test, y_test)

    if SAVE_MODEL is True:
        model.save_model(classifier=classifier, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
