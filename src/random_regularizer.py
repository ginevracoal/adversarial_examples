# -*- coding: utf-8 -*-

from adversarial_classifier import *
from art.classifiers import TFClassifier
import tensorflow as tf
from utils import *
import sys


class RandomRegularizer(TFClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, sess):
        # Define dataset infos
        self.dataset_name = dataset_name
        self.data_format = data_format
        self.num_classes = num_classes
        self._input_shape = input_shape
        # Define input and output placeholders
        self.input_ph = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]])  # 28, 28, 3
        self.output_ph = tf.placeholder(tf.int32, shape=[None, num_classes])
        # Define the architecture
        self.batch_size, self.epochs = self._set_training_params()
        self.train_op = self._set_train_op()
        self.trained = False
        # TF Classifier
        super(RandomRegularizer, self).__init__(clip_values=(0, 1), input_ph=self.input_ph, logits=self.logits,
                                     output_ph=self.output_ph, train=self.train_op, loss=self.loss, learning=None, sess=sess)

    def _set_training_params(self):
        if self.dataset_name == "mnist":
            return 128, 12
        elif self.dataset_name == "cifar":
            return 128, 120

    def _set_train_op(self):
        if self.dataset_name == "mnist":
            x = tf.layers.conv2d(inputs=self.input_ph, filters=32, kernel_size=(3,3), activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer= tf.constant_initializer())
            x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3,3), activation=tf.nn.relu)
            x = tf.layers.max_pooling2d(inputs=x, pool_size=(2,2),
                                        strides=[self.input_shape[0], self.input_shape[1]])
            x = tf.layers.dropout(inputs=x, rate=0.25)#training=mode == tf.estimator.ModeKeys.TRAIN)
            x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
            x = tf.layers.dropout(inputs=x, rate=0.5)#training=mode == tf.estimator.ModeKeys.TRAIN)
            x = tf.contrib.layers.flatten(x)
            self.logits = tf.layers.dense(inputs=x, units=self.num_classes, activation=tf.nn.softmax)

            # Train operator
            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self.logits, onehot_labels=self.output_ph))
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(self.loss)

            return train_op

            #model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

        #elif self.dataset_name

    def loss(self):
        raise NotImplementedError

    def train(self, x_train, y_train):
        print("\nTraining infos:\nbatch_size = ", self.batch_size, "\nepochs =", self.epochs,
              "\nx_train.shape = ", x_train.shape, "\ny_train.shape = ", y_train.shape, "\n")

        start_time = time.time()
        self.fit(x_train, y_train, batch_size=self.batch_size, nb_epochs=self.epochs)
        print("\nTraining time: --- %s seconds ---" % (time.time() - start_time))

        self.trained = True
        return self

    def save_model(self, classifier, model_name):
        """
        Saves the trained model and adds the current datetime to the filename.
        Example of saved model: `trained_models/2019-05-20/baseline.h5`

        :param classifier: trained classifier
        :param model_name: name of the model
        """
        if self.trained:
            classifier.save(filename=model_name+".h5", path=RESULTS + time.strftime('%Y-%m-%d') + "/")

    def load_classifier(self, relative_path, sess):
        """
        Loads a pretrained classifier.
        :param relative_path: is the relative path w.r.t. trained_models folder, `2019-05-20/baseline.h5` in the example
        from save_model()
        :param sess: tf session
        returns: trained classifier
        """
        print("\nLoading model:", str(relative_path))
        # load a trained model
        model = load_model(relative_path)
        classifier = TFClassifier(clip_values=(0, 1), input_ph=model.input_ph, logits=model.logits, sess=sess,
                                  output_ph=model.output_ph, train=model.train, loss=model.loss, learning=None)
        return classifier

    def evaluate_test(self, classifier, x_test, y_test):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        :return: x_test predictions
        """
        print("\n===== Test set evaluation =====")
        print("\nTesting infos:\nx_test.shape = ", x_test.shape, "\ny_test.shape = ", y_test.shape, "\n")

        y_test_pred = np.argmax(classifier.predict(x_test), axis=1)
        y_test_true = np.argmax(y_test, axis=1)
        correct_preds = np.sum(y_test_pred == np.argmax(y_test, axis=1))

        print("Correctly classified: {}".format(correct_preds))
        print("Incorrectly classified: {}".format(len(x_test) - correct_preds))

        acc = np.sum(y_test_pred == y_test_true) / y_test.shape[0]
        print("Test accuracy: %.2f%%" % (acc * 100))

        # classification report over single classes
        print(classification_report(np.argmax(y_test, axis=1), y_test_pred, labels=range(self.num_classes)))

        return y_test_pred


def main(dataset_name, test, attack):

    # load dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name, test=test)

    # Tensorflow session and initialization
    sess = tf.Session()
    # Define the tensorflow graph
    randReg = RandomRegularizer(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                dataset_name=dataset_name, sess=sess)
    sess.run(tf.global_variables_initializer())

    classifier = randReg.train(x_train, y_train)
    randReg.save_model(classifier=classifier, model_name=dataset_name + "_" + attack + "_baseline")

    # evaluations #
    randReg.evaluate_test(classifier=classifier, x_test=x_test, y_test=y_test)


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        attack = sys.argv[3]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        attack = input("\nChoose an attack ("+ATTACKS+"): ")

    main(dataset_name=dataset_name, test=test, attack=attack)
