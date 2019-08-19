# -*- coding: utf-8 -*-

from adversarial_classifier import *
from art.classifiers import TFClassifier
import keras.backend as K
import tensorflow as tf
from utils import *
import sys
import random
from baseline_convnet import BaselineConvnet
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier

# todo: docstrings!
# todo: unittest


class RandomRegularizerK(KerasClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, sess, lam):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        self.sess=sess
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.data_format = data_format
        self.lam = lam
        self.batch_size, self.epochs = self._set_training_params()
        self.model = self._set_layers()
        super(RandomRegularizerK, self).__init__(build_fn=self._set_layers)

    def _set_training_params(self):
        if self.dataset_name == "mnist":
            return 100, 12
        elif self.dataset_name == "cifar":
            return 100, 120

    def _get_logits(self, inputs):
        if self.dataset_name == "mnist":
            x = Conv2D(32, kernel_size=(3, 3), activation='relu', data_format=self.data_format)(inputs)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            logits = Dense(self.num_classes, activation='softmax')(x)
            return logits

    def _set_layers(self):
        inputs = Input(shape=self.input_shape)
        logits = self._get_logits(inputs=inputs)
        model = Model(inputs=inputs, outputs=logits)
        model.compile(loss=self.loss_wrapper(inputs), optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
        # model.summary()
        return model

    def loss_wrapper(self, inputs):
        def custom_loss(y_true, y_pred):
            return K.binary_crossentropy(y_true, y_pred) + self.lam * self.regularizer(inputs=inputs, y_true=y_true)
        return custom_loss

    def regularizer(self, inputs, y_true, max_nproj=6):
        """
        This regularization term penalizes the expected value of loss gradient, evaluated on random projections of the
        input points.
        :param inputs:
        :param y_true:
        :param max_nproj:
        :return: regularization term for the objective loss function.
        """
        rows, cols, channels = self.input_shape
        flat_images = tf.reshape(inputs,shape=[self.batch_size, rows * cols * channels])
        n_proj = 1  # random.randint(3, max_proj)
        n_features = rows * cols * channels
        inverse_projections = np.empty(shape=(n_proj,self.batch_size,n_features))
        for proj in range(n_proj):
            size = random.randint(1,20)
            seed = random.randint(1,100)
            n_components = size*size*channels
            print("\nproj =", proj, ", size =", size, ", seed =", seed)
            projector = GaussianRandomProjection(n_components=n_components, random_state=seed)
            proj_matrix = np.float32(projector._make_random_matrix(n_components,n_features))
            pinv = np.linalg.pinv(proj_matrix)
            projections = tf.matmul(a=flat_images, b=proj_matrix, transpose_b=True)
            # todo: inverse_projections cantains all the projections for the whole datset from the chosen projector.
            # todo: instead I would like to use a different projection on each possible point...
            inverse_projections = tf.matmul(a=projections, b=pinv, transpose_b=True)
            inverse_projections = tf.reshape(inverse_projections, shape=tf.TensorShape([self.batch_size, rows,cols,channels]))
            #print(proj_matrix.shape)
            #print(pinv.shape)

        proj_logits = self._get_logits(inputs=inverse_projections)
        #print(proj_logits) # Tensor("loss/dense_2_loss/dense_4/Softmax:0", shape=(100, 10), dtype=float32)

        loss = K.categorical_crossentropy(target=y_true, output=proj_logits)
        #print(loss) #Tensor("loss/dense_2_loss/Neg:0", shape=(100,), dtype=float32)

        loss_gradient = K.gradients(loss=loss, variables=inputs) #  shape=(100, 28, 28, 1) dtype=float32>]
        regularization = tf.reduce_sum(tf.square(tf.norm(loss_gradient, ord=2, axis=0))) / (n_proj*self.batch_size)
        return regularization

    def compute_projections(self):
        return None

    def train(self, x_train, y_train):
        print("\nTraining infos:\nbatch_size = ", self.batch_size, "\nepochs = ", self.epochs,
              "\nx_train.shape = ", x_train.shape, "\ny_train.shape = ", y_train.shape, "\n")

        start_time = time.time()
        self.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
        print("\nTraining time: --- %s seconds ---" % (time.time() - start_time))

        self.trained = True
        return self

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

        y_test_pred = self.predict(x_test)# np.argmax(self.predict(x_test))#, axis=1)
        y_test_true = np.argmax(y_test, axis=1)
        correct_preds = np.sum(y_test_pred == np.argmax(y_test, axis=1))

        print("Correctly classified: {}".format(correct_preds))
        print("Incorrectly classified: {}".format(len(x_test) - correct_preds))

        acc = np.sum(y_test_pred == y_test_true) / y_test.shape[0]
        print("Test accuracy: %.2f%%" % (acc * 100))

        # classification report over single classes
        print(classification_report(np.argmax(y_test, axis=1), y_test_pred, labels=range(self.num_classes)))

        return y_test_pred

###################################################

class RandomRegularizerTF(TFClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, lam, sess):
        # Define dataset infos
        self.sess = sess
        #self.x_train = x_train
        self._input_shape = input_shape
        self.num_classes = num_classes
        self.data_format = data_format
        self.dataset_name = dataset_name
        self.lam = lam  # regularization weight
        # Define the architecture
        self.batch_size, self.epochs = self._set_training_params()
        #self.trained = False

        # Define input and output placeholders
        self.input_ph = tf.placeholder(tf.float32, shape=[self.batch_size, input_shape[0], input_shape[1], input_shape[2]])
        self.output_ph = tf.placeholder(tf.int32, shape=[self.batch_size, num_classes])
        #train_op = self._train_op()
        self.logits = self._get_logits(self.input_ph)
        # TF Classifier
        super(RandomRegularizerTF, self).__init__(clip_values=(0, 1), input_ph=self.input_ph, logits=self.logits,
                                                  output_ph=self.output_ph, train=self._train_op(),
                                                  loss=self.loss_wrapper(self.input_ph),
                                                  learning=None, sess=sess)

    def _set_training_params(self):
        if self.dataset_name == "mnist":
            return 128, 12
        elif self.dataset_name == "cifar":
            return 128, 120

    def _get_logits_tf(self, inputs):
        if self.dataset_name == "mnist":
            x = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=(3,3), activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                 bias_initializer= tf.constant_initializer())
            x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(3,3), activation=tf.nn.relu)
            x = tf.layers.max_pooling2d(inputs=x, pool_size=(2,2),
                                        strides=[self._input_shape[0], self._input_shape[1]])
            x = tf.layers.dropout(inputs=x, rate=0.25) #training=mode == tf.estimator.ModeKeys.TRAIN)
            x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
            x = tf.layers.dropout(inputs=x, rate=0.5) #training=mode == tf.estimator.ModeKeys.TRAIN)
            x = tf.contrib.layers.flatten(x)
            logits = tf.layers.dense(inputs=x, units=self.num_classes, activation=tf.nn.softmax)
            return logits

        # elif self.dataset_name == "cifar":
            # todo: implement cifar architecture

    def _get_logits(self, inputs):
        if self.dataset_name == "mnist":
            x = Conv2D(32, kernel_size=(3, 3), activation='relu', data_format=self.data_format)(inputs)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            logits = Dense(self.num_classes, activation='softmax')(x)
            return logits

    def loss_wrapper(self, inputs):
        def custom_loss(y_true, y_pred):
            return K.binary_crossentropy(y_true, y_pred) + self.lam * self.regularizer(inputs=inputs, labels=y_true)
        return custom_loss

    def _train_op(self):
        inputs = Input(shape=self.input_shape)
        logits = self._get_logits(inputs=inputs)
        # todo: non ho capito se i logits così si aggiornano ad ogni step oppure no...

        #self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=self.output_ph))
        #self.loss = self.proj_loss(inputs=self.input_ph, logits=logits, labels=self.output_ph)

        #optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        #train_operator = optimizer.minimize(self.loss)
        #return train_operator

        model = Model(inputs=inputs, outputs=logits)
        model.compile(loss=self.loss_wrapper(inputs), optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
        # model.summary()
        return model

    # currently unused
    def project_image(self, im, size, channels, projector, psinverse):
        #projection = np.empty(shape=(size*size, channels))
        projected_image = np.empty(shape=(size, size, channels))
        projection = projector.fit_transform(im)  # .reshape(1, -1))#.reshape(size, size)#, channels)
        exit()
        for channel in range(channels):
            ch_im = tf.reshape(im[:,:, channel], shape=[1,-1])

            projection = projector.fit_transform(im[:,:, channel])#.reshape(1, -1))#.reshape(size, size)#, channels)
            print(projection)
            exit()
            projection = tf.reshape(projection, shape=[size,size])
            # apply pseudoinverse and get full dimensional projected data
            projected_image[:, channel] = map(psinverse.dot, projection)
        return projected_image

    def regularizer(self, inputs, labels, max_nproj=6):
        """
        This regularization term penalizes the expected value of loss gradient, evaluated on random projections of the
        input points.
        :param inputs:
        :param labels:
        :param max_nproj:
        :return: regularization term for the objective loss function.
        """
        rows, cols, channels = self.input_shape
        flat_images = tf.reshape(inputs,shape=[self.batch_size, rows * cols * channels])
        n_proj = 1  # random.randint(3, max_proj)
        n_features = rows * cols * channels
        inverse_projections = np.empty(shape=(n_proj,self.batch_size,n_features))
        for proj in range(n_proj):
            size = random.randint(1,20)
            seed = random.randint(1,100)
            n_components = size*size*channels
            print("\nproj =", proj, ", size =", size, ", seed =", seed)
            projector = GaussianRandomProjection(n_components=n_components, random_state=seed)
            proj_matrix = np.float32(projector._make_random_matrix(n_components,n_features))
            pinv = np.linalg.pinv(proj_matrix)
            projections = tf.matmul(a=flat_images, b=proj_matrix, transpose_b=True)
            # todo: inverse_projections cantains all the projections for the whole datset from the chosen projector.
            # todo: instead I would like to use a different projection on each possible point...
            inverse_projections = tf.matmul(a=projections, b=pinv, transpose_b=True)
            inverse_projections = tf.reshape(inverse_projections, shape=tf.TensorShape([self.batch_size, rows,cols,channels]))
            #print(proj_matrix.shape)
            #print(pinv.shape)

        # todo: ogni volta calcolo il regolarizzatore su un batch diverso

        #loss_gradient = self.loss_gradient(x=inverse_projections, y=labels)
        loss = K.mean(K.categorical_crossentropy(target=labels, output=self._get_logits(inputs=inverse_projections)))
        loss_gradient = K.gradients(loss=loss,variables=inverse_projections)
        reg = tf.norm(loss_gradient, ord=2, axis=1)
        return reg

    def train(self, x_train, y_train):
        print("\nTraining infos:\nbatch_size = ", self.batch_size, "\nepochs = ", self.epochs,
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


def main(dataset_name, test, attack, lam):
    # load dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name, test=test)

    # Tensorflow session and initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Define the tensorflow graph
    randReg = RandomRegularizerK(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                  dataset_name=dataset_name, sess=sess, lam=lam)
    classifier = randReg.train(x_train, y_train)#, batch_size=randReg.batch_size, epochs=randReg.epochs)
    #randReg.save_model(classifier=classifier, model_name=dataset_name + "_" + attack + "_baseline")

    # evaluations #
    randReg.evaluate_test(classifier=classifier, x_test=x_test, y_test=y_test)

if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        attack = sys.argv[3]
        lam = float(sys.argv[4])

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        attack = input("\nChoose an attack ("+ATTACKS+"): ")
        lam = input("\nChoose lambda regularization weight.")

    main(dataset_name=dataset_name, test=test, attack=attack, lam=lam)
