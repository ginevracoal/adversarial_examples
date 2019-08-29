# -*- coding: utf-8 -*-

import random
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization
from art.classifiers import KerasClassifier as artKerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier as sklKerasClassifier
import keras.losses
from random_ensemble import *

# todo: docstrings
# todo: unittest

PROJ_MODE = "grayscale, channels, projected_loss"


class RandomRegularizer(sklKerasClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, sess, lam, projection_mode, n_proj):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        self.sess=sess
        self.input_shape = input_shape
        self.n_proj = n_proj
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.data_format = data_format
        self.lam = lam
        self.projection_mode = projection_mode
        self._set_training_params()
        self._set_layers()
        self.inputs = Input(shape=self.input_shape)
        print("\nbatch_size=",self.batch_size,"\nepochs=",self.epochs,"\nn_proj=",self.n_proj)
        super(RandomRegularizer, self).__init__(build_fn=self.model, batch_size=self.batch_size, epochs=self.epochs)

    def _set_training_params(self):
        if self.dataset_name == "mnist":
            self.epochs = 12
            if test:
                self.batch_size = 100
            else:
                self.batch_size = 1000
        elif self.dataset_name == "cifar":
            self.epochs = 60
            if test:
                self.batch_size = 100
            else:
                self.batch_size = 1000

    def _get_logits(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
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

        elif self.dataset_name == "cifar":
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
            x = BatchNormalization()(x)
            x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.2)(x)
            x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.3)(x)
            x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.4)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            logits = Dense(10, activation='softmax')(x)
            return logits

    def _set_layers(self):
        inputs = Input(shape=self.input_shape)
        logits = self._get_logits(inputs=inputs)
        self.outputs = logits
        self.model = Model(inputs=inputs, outputs=logits)

    def loss_wrapper(self, inputs, outputs):
        """ Loss wrapper for custom loss function.
        # todo: docstring and test
        """
        def custom_loss(y_true, y_pred):
            if self.projection_mode == "grayscale":
                print("\ngrayscale regularization.")
                return K.binary_crossentropy(y_true, y_pred) + self.lam * self.grayscale_regularizer(inputs=inputs, outputs=outputs)
            elif self.projection_mode == "channels":
                print("\nchannels regularization.")
                return K.binary_crossentropy(y_true, y_pred) + self.lam * self.channels_regularizer(inputs=inputs, outputs=outputs)
            elif self.projection_mode == "projected_loss":
                print("\nprojected_loss regularization.")
                return K.binary_crossentropy(y_true, y_pred) + self.lam * self.projected_loss_regularizer(inputs=inputs, outputs=outputs)
            else:
                raise NotImplementedError("Wrong projection mode. Supported modes: "+PROJ_MODE)
        return custom_loss

    def _get_loss_gradient(self, inputs, outputs):#flat_inputs, inverse_projection, outputs, project_points=True):
        # todo: docstring & unittest

        axis = 1 if self.data_format == "channels_first" else -1

        def _compute_gradients(tensor, var_list):
            grads = tf.gradients(tensor, var_list)
            return [grad if grad is not None else tf.zeros_like(var)
                    for var, grad in zip(var_list, grads)]

        loss = K.categorical_crossentropy(target=outputs, output=self._get_logits(inputs=inputs),
                                          from_logits=True, axis=axis)
        loss_gradient = _compute_gradients(loss, [inputs])[0]
        return loss_gradient

    def grayscale_regularizer(self, inputs, outputs):
        # todo: docstring & unittest

        regularization = 0

        size = random.randint(6, 15)
        seed = random.randint(1, 100)  # one different seed for projecting each batch
        print("\nsize =", size, ", seed =", seed)

        _, rows, cols, channels = self.inputs.get_shape().as_list()
        inputs = tf.cast(inputs, tf.float32)
        flat_images = tf.reshape(inputs, shape=[self.batch_size, rows * cols * channels])

        if channels == 3:
            grayscale_images = tf.image.rgb_to_grayscale(inputs)
            flat_images = tf.reshape(tensor=grayscale_images, shape=[self.batch_size, rows * cols * 1])
            channels = 1

        n_features = rows * cols * channels
        n_components = size * size * channels

        projector = GaussianRandomProjection(n_components=n_components, random_state=seed)
        proj_matrix = np.float32(projector._make_random_matrix(n_components, n_features))
        pinv = np.linalg.pinv(proj_matrix)
        projections = tf.matmul(a=flat_images, b=proj_matrix, transpose_b=True)
        inverse_projections = tf.matmul(a=projections, b=pinv, transpose_b=True)
        inverse_projections = tf.reshape(inverse_projections,
                                         shape=tf.TensorShape([self.batch_size, rows, cols, channels]))

        loss_gradient = self._get_loss_gradient(inputs=inverse_projections, outputs=outputs)
        regularization += tf.reduce_sum(tf.math.square(tf.norm(loss_gradient, ord=2, axis=1)))

        return regularization / self.batch_size*self.n_proj

    def channels_regularizer(self, inputs, outputs):
        # todo: docstring & unittest

        regularization = 0

        size = random.randint(18, 28)
        seed = random.randint(1, 100)  # one different seed for projecting each batch
        print("\nsize =", size, ", seed =", seed)

        _, rows, cols, channels = self.inputs.get_shape().as_list()
        inputs = tf.cast(inputs, tf.float32)

        # projection matrices
        n_features = rows * cols
        n_components = size * size
        projector = GaussianRandomProjection(n_components=n_components, random_state=seed)
        proj_matrix = np.float32(projector._make_random_matrix(n_components, n_features))
        pinv = np.linalg.pinv(proj_matrix)

        # channel projections
        for channel in range(channels):
            flat_images = tf.reshape(tensor=inputs[:, :, :, channel], shape=(self.batch_size, rows * cols))

            # project
            channel_projection = tf.matmul(a=flat_images, b=proj_matrix, transpose_b=True)
            channel_inverse_projection = tf.matmul(a=channel_projection, b=pinv, transpose_b=True)
            channel_inverse_projection = tf.reshape(channel_inverse_projection, shape=(self.batch_size, rows, cols, 1))

            loss_gradient = self._get_loss_gradient(inputs=channel_inverse_projection, outputs=outputs)
            regularization += tf.reduce_sum(tf.math.square(tf.norm(loss_gradient, ord=2, axis=1)))

        return regularization / self.batch_size*self.n_proj

    def projected_loss_regularizer(self, inputs, outputs):
        # todo: docstring & unittest

        # set params
        size = random.randint(8, 28)
        seed = random.randint(1, 100)
        print("\nsize =", size, ", seed =", seed)
        _, rows, cols, channels = self.inputs.get_shape().as_list()
        n_features = rows * cols * channels
        n_components = size * size * channels

        # compute loss gradient
        # flat_images = tf.reshape(tf.cast(inputs, tf.float32), shape=[self.batch_size, n_features])
        inputs = tf.cast(inputs, tf.float32)
        loss_gradient = self._get_loss_gradient(inputs=inputs, outputs=outputs)
                                                # flat_inputs=flat_images, inverse_projection=None,
                                                # outputs=outputs, project_points=False)

        # project loss gradient
        projector = GaussianRandomProjection(n_components=n_components, random_state=seed)
        proj_matrix = np.float32(projector._make_random_matrix(n_components, n_features))
        flat_loss_gradient = tf.reshape(loss_gradient, shape=[self.batch_size, n_features])
        projected_loss = tf.matmul(a=flat_loss_gradient, b=proj_matrix, transpose_b=True)

        # compute regularization term
        projected_loss = tf.reshape(projected_loss, shape=(self.batch_size, size, size, channels))
        regularization = tf.reduce_sum(tf.math.square(tf.norm(projected_loss, ord=2, axis=1)))

        return regularization / self.batch_size*self.n_proj

    def train(self, x_train, y_train):
        print("\nTraining infos:\nbatch_size = ", self.batch_size, "\nepochs = ", self.epochs,
              "\nx_train.shape = ", x_train.shape, "\ny_train.shape = ", y_train.shape, "\n")

        self.batches = int(len(x_train)/self.batch_size)
        x_train_batches = np.split(x_train, self.batches)
        y_train_batches = np.split(y_train, self.batches)

        start_time = time.time()
        for batch in range(self.batches):
            print("\n=== training batch", batch+1,"/",self.batches,"===")
            inputs = tf.convert_to_tensor(x_train_batches[batch])
            outputs = tf.convert_to_tensor(y_train_batches[batch])
            early_stopping = keras.callbacks.EarlyStopping(monitor='loss', verbose=1)

            for proj in range(self.n_proj):
                print("\nprojection",proj+1,"/",self.n_proj)
                loss = self.loss_wrapper(inputs,outputs)
                self.model.compile(loss=loss, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
                self.model.fit(x_train_batches[batch], y_train_batches[batch], epochs=self.epochs, batch_size=self.batch_size,
                               callbacks=[early_stopping])  #callbacks=[EpochIdxCallback(self.model)]

        print("\nTraining time: --- %s seconds ---" % (time.time() - start_time))
        return self

    def evaluate_test(self, x_test, y_test):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        :return: x_test predictions
        """

        # todo: set classes_ attribute for loaded models
        if len(y_test.shape) == 2 and y_test.shape[1] > 1:
            self.classes_ = np.arange(y_test.shape[1])
        elif (len(y_test.shape) == 2 and y_test.shape[1] == 1) or len(y_test.shape) == 1:
            self.classes_ = np.unique(y_test)

        print("\n===== Test set evaluation =====")
        print("\nTesting infos:\nx_test.shape = ", x_test.shape, "\ny_test.shape = ", y_test.shape, "\n")

        y_test_pred = self.predict(x_test) # self.predict
        y_test_true = np.argmax(y_test, axis=1)
        correct_preds = np.sum(y_test_pred == np.argmax(y_test, axis=1))

        print("Correctly classified: {}".format(correct_preds))
        print("Incorrectly classified: {}".format(len(x_test) - correct_preds))

        acc = np.sum(y_test_pred == y_test_true) / y_test.shape[0]
        print("Test accuracy: %.2f%%" % (acc * 100))

        # classification report over single classes
        print(classification_report(np.argmax(y_test, axis=1), y_test_pred, labels=range(self.num_classes)))

        return y_test_pred

    def evaluate_adversaries(self, x_test, y_test, method, dataset_name, adversaries_path=None, test=False):
        print("\n===== Adversarial evaluation =====")

        if len(y_test.shape) == 2 and y_test.shape[1] > 1:
            self.classes_ = np.arange(y_test.shape[1])
        elif (len(y_test.shape) == 2 and y_test.shape[1] == 1) or len(y_test.shape) == 1:
            self.classes_ = np.unique(y_test)

        # generate adversaries on the test set
        x_test_adv = self._get_adversaries(self, x_test, y_test, method=method, dataset_name=dataset_name,
                                           adversaries_path=adversaries_path, test=test)
        # debug
        # print(len(x_test), len(x_test_adv))

        # evaluate the performance on the adversarial test set
        y_test_adv = self.predict(x_test_adv)
        y_test_true = np.argmax(y_test, axis=1)
        nb_correct_adv_pred = np.sum(y_test_adv == y_test_true)

        print("\nAdversarial test data.")
        print("Correctly classified: {}".format(nb_correct_adv_pred))
        print("Incorrectly classified: {}".format(len(x_test_adv) - nb_correct_adv_pred))

        acc = nb_correct_adv_pred / y_test.shape[0]
        print("Adversarial accuracy: %.2f%%" % (acc * 100))

        # classification report
        print(classification_report(np.argmax(y_test, axis=1), y_test_adv, labels=range(self.num_classes)))
        return x_test_adv, y_test_adv

    def _get_adversaries(self, trained_classifier, x, y, test, method, dataset_name, adversaries_path):
        """
        Generates adversaries on the input data x using a given method or loads saved data if available.

        :param classifier: trained classifier
        :param x: input data
        :param method: art.attack method
        :param adversaries_path: path of saved pickle data
        :return: adversarially perturbed data
        """
        if adversaries_path is None:
            classifier = artKerasClassifier((MIN, MAX), trained_classifier.model, use_logits=False)
            classifier._loss = self.loss_wrapper(tf.convert_to_tensor(x), None)
            classifier.custom_loss = self.loss_wrapper(tf.convert_to_tensor(x), None)
            print("\nGenerating adversaries with", method, "method on", dataset_name)
            x_adv = None
            if method == 'fgsm':
                attacker = FastGradientMethod(classifier, eps=0.5)
                x_adv = attacker.generate(x=x)
            elif method == 'deepfool':
                attacker = DeepFool(classifier)
                x_adv = attacker.generate(x)
            elif method == 'virtual':
                attacker = VirtualAdversarialMethod(classifier)
                x_adv = attacker.generate(x)
            elif method == 'carlini_l2':
                attacker = CarliniL2Method(classifier, targeted=False)
                x_adv = attacker.generate(x=x, y=y)
            elif method == 'carlini_linf':
                attacker = CarliniLInfMethod(classifier, targeted=False)
                x_adv = attacker.generate(x=x, y=y)
            elif method == 'pgd':
                attacker = ProjectedGradientDescent(classifier)
                x_adv = attacker.generate(x=x)
            elif method == 'newtonfool':
                attacker = NewtonFool(classifier)
                x_adv = attacker.generate(x=x)
        else:
            print("\nLoading adversaries generated with", method, "method on", dataset_name)
            x_adv = load_from_pickle(path=adversaries_path, test=test)  # [0]

        if test:
            return x_adv[:TEST_SIZE]
        else:
            return x_adv

# class EpochIdxCallback(keras.callbacks.Callback):
#     def __init__(self, model):
#         super(EpochIdxCallback,self).__init__()
#         self.model = model
#     def on_epoch_begin(self, epoch, logs={}):
#         self.epoch = epoch
#         return epoch
#     def set_model(self, model):
#         return self.model


def main(dataset_name, test, lam, projection_mode, n_proj):

    # Tensorflow session and initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    os.makedirs(os.path.dirname(RESULTS+time.strftime('%Y-%m-%d')+"/"), exist_ok=True)

    # load dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name, test=test)

    classifier = RandomRegularizer(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                   dataset_name=dataset_name, sess=sess, lam=lam, projection_mode=projection_mode,
                                   n_proj=n_proj)

    modelname = str(dataset_name) + "_randreg_lam=" + str(classifier.lam) + \
                "_batch="+str(classifier.batch_size)+"_epochs="+str(classifier.epochs)+"_proj="+str(classifier.n_proj)+\
                "_"+str(classifier.projection_mode)+".h5"
    model_path = RESULTS+time.strftime('%Y-%m-%d') +"/"+ modelname
    # model_path = TRAINED_MODELS+"random_regularizer/" + modelname

    classifier.train(x_train, y_train)
    classifier.model.save_weights(model_path)

    # evaluations #
    classifier.evaluate_test(x_test=x_test, y_test=y_test)

    ######### todo: bug on adversarial evaluation. It only works on loaded models..
    del classifier
    classifier = RandomRegularizer(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                   dataset_name=dataset_name, sess=sess, lam=lam, projection_mode=projection_mode,
                                   n_proj=n_proj)

    classifier.model.load_weights(model_path)

    print("\nBaseline adversaries")
    for attack in ['fgsm','pgd','deepfool','carlini_linf']:
        filename = dataset_name + "_x_test_" + attack + ".pkl"
        x_test_adv = classifier.evaluate_adversaries(x_test, y_test, method=attack, dataset_name=dataset_name,
                                                     adversaries_path=DATA_PATH+filename,
                                                     test=test)

    print("\nRandreg adversaries")
    for attack in ['fgsm','pgd','deepfool','carlini_linf']:
        filename = dataset_name + "_x_test_" + attack + "_randreg.pkl"
        x_test_adv = classifier.evaluate_adversaries(x_test, y_test, method=attack, dataset_name=dataset_name,
                                                     #adversaries_path=DATA_PATH+filename,
                                                     test=test)
        # save_to_pickle(data=x_test_adv, filename=filename)


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        n_proj = int(sys.argv[3])
        lam = float(sys.argv[4])
        projection_mode = sys.argv[5]

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        n_proj = int(input("\nChoose the number of projections (type=int): "))
        lam = float(input("\nChoose lambda regularization weight (type=float): "))
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")

    main(dataset_name=dataset_name, test=test, lam=lam, projection_mode=projection_mode, n_proj=n_proj)
