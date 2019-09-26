# -*- coding: utf-8 -*-

import random
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization
from art.classifiers import KerasClassifier as artKerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier as sklKerasClassifier
import keras.losses
from random_ensemble import *
from projection_functions import *
from adversarial_classifier import AdversarialClassifier as myAdvClassifier

DERIVATIVES_ON_INPUTS = True  # if True compute gradient derivatives w.r.t. the inputs, else w.r.t. the projected inputs
TRAINED_MODELS = "../trained_models/random_regularizer/"

L_RATE = 5
MIN_SIZE = 2
MAX_SIZE = 4
MIN_PROJ = 1
MAX_PROJ = 3
TEST_SIZE = 2
TEST_PROJ = 1
PROJ_MODE = "no_projections, loss_on_projections, projected_loss, loss_on_perturbations"
CHANNEL_MODE = "grayscale" # "channels, grayscale"

class RandomRegularizer(sklKerasClassifier):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, sess, lam, projection_mode, eps, test=False):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        self.sess=sess
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.data_format = data_format
        self.lam = lam
        self.projection_mode = projection_mode
        self._set_training_params(test)
        self.test = test
        self._set_model()
        self.n_proj = None
        self.inputs = Input(shape=self.input_shape)
        self.adversarial_classifier = myAdvClassifier(input_shape=self.input_shape, num_classes=self.num_classes,
                                                            data_format=self.data_format, eps=eps)
        print("\nprojection mode =", self.projection_mode, "channel mode =", CHANNEL_MODE,", lambda =", self.lam)
        print("\nbatch_size =",self.batch_size,", epochs =",self.epochs,", lr =",L_RATE)
        print("\nn_proj~(",MIN_PROJ,",",MAX_PROJ,"), size_proj~(",MIN_SIZE,",",MAX_SIZE,")")
        super(RandomRegularizer, self).__init__(build_fn=self.model, batch_size=self.batch_size, epochs=self.epochs)

    def _set_training_params(self, test):
        if test:
            self.epochs = 1
            self.batch_size = 100
        else:
            if self.dataset_name == "mnist":
                self.epochs = 12
                self.batch_size = 1000
            elif self.dataset_name == "cifar":
                self.epochs = 12
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

    def _set_model(self):
        inputs = Input(shape=self.input_shape)
        outputs = self._get_logits(inputs=inputs)
        self.model = Model(inputs=inputs, outputs=outputs)

    def loss_wrapper(self, inputs, outputs):
        """ Loss wrapper for custom loss function.
        :param inputs: input data for the loss, type=tf.tensor, shape=(batch_size, rows, cols, channels)
        :param outputs: output data for the loss, type=tf.tensor, shape=(batch_size, n_classes)
        """
        channels = inputs.get_shape().as_list()[3]
        inputs = tf.cast(inputs, tf.float32)

        if CHANNEL_MODE == "grayscale" and channels == 3:
            inputs = tf.image.rgb_to_grayscale(inputs)

        def custom_loss(y_true, y_pred):
            if self.projection_mode == "no_projections":
                regularization_term = self._no_projections_regularizer(inputs=inputs, outputs=outputs)
            elif self.projection_mode == "loss_on_projections":
                regularization_term = self._loss_on_projections_regularizer(inputs=inputs, outputs=outputs)
            elif self.projection_mode == "projected_loss":
                regularization_term = self._projected_loss_regularizer(inputs=inputs, outputs=outputs)
            elif self.projection_mode == "loss_on_perturbations":
                regularization_term = self._loss_on_perturbations_regularizer(inputs=inputs, outputs=outputs)
            else:
                raise NotImplementedError("Wrong projection mode. Supported modes: "+PROJ_MODE)

            return K.binary_crossentropy(y_true, y_pred) + self.lam * regularization_term

        return custom_loss

    def _compute_gradients(self, tensor, var_list):
        """
        Handles None values in tensors gradient computation.
        :param tensor: input tensor to be derivated, type=tensor, shape=(batch_size,)
        :param var_list: list of variables on which to compute derivatives, shape=(batch_size, rows, cols, channels)
        :return: tf.gradients on tensor w.r.t. var_list
        """
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var)
                for var, grad in zip(var_list, grads)]

    def _get_n_channels(self, inputs, channel_mode=CHANNEL_MODE):
        # if channel_mode == "channels":
        #     return inputs.get_shape().as_list()[3]
        # elif channel_mode == "one_channel":
        #     return 1
        # else:
        #     raise ValueError("Projection mode not supported. Choose between:", CHANNEL_MODE)
        return inputs.get_shape().as_list()[3]

    def _no_projections_regularizer(self, inputs, outputs):
        if DERIVATIVES_ON_INPUTS is False:
            raise AttributeError("\nYou can only compute derivatives on the inputs. Set DERIVATIVES_ON_INPUTS = True.")

        size = random.randint(MIN_SIZE, MAX_SIZE) if self.test is False else TEST_SIZE
        seed = random.randint(1, 100)
        print("\nsize =", size, ", seed =", seed)
        axis = 1 if self.data_format == "channels_first" else -1

        regularization = 0
        channels = self._get_n_channels(inputs)

        for channel in range(channels):
            channel_inputs = tf.expand_dims(input=inputs[:, :, :, channel], axis=3)

            loss = K.categorical_crossentropy(target=outputs, output=self._get_logits(inputs=channel_inputs),
                                              from_logits=True, axis=axis)
            loss_gradient = self._compute_gradients(loss, [channel_inputs])[0]
            regularization += tf.reduce_sum(tf.math.square(tf.norm(loss_gradient / channels, ord=2, axis=0)))

        return regularization / self.batch_size

    def _loss_on_projections_regularizer(self, inputs, outputs):
        size = random.randint(MIN_SIZE, MAX_SIZE) if self.test is False else TEST_SIZE
        seed = random.randint(1, 100)
        print("\nsize =", size, ", seed =", seed)

        channels = self._get_n_channels(inputs)

        axis = 1 if self.data_format == "channels_first" else -1
        regularization = 0
        for channel in range(channels):
            channel_inputs = tf.expand_dims(input=inputs[:, :, :, channel], axis=3)

            if DERIVATIVES_ON_INPUTS:
                loss = K.categorical_crossentropy(target=outputs,
                                                  output=self._get_logits(
                                                      inputs=tf_flat_projection(input_data=channel_inputs,
                                                                                random_seed=seed, size_proj=size)[1]),
                                                  from_logits=True, axis=axis)
                loss_gradient = self._compute_gradients(loss, [channel_inputs])[0]
            else:
                pinv_channel_inputs = tf_flat_projection(input_data=channel_inputs, random_seed=seed, size_proj=size)[1]
                loss = K.categorical_crossentropy(target=outputs, output=self._get_logits(inputs=pinv_channel_inputs),
                                                  from_logits=True, axis=axis)
                loss_gradient = self._compute_gradients(loss, [pinv_channel_inputs])[0]

            # expectation #
            regularization += tf.reduce_sum(tf.math.square(tf.norm(loss_gradient/channels, ord=2, axis=0)))
        return regularization / (self.batch_size * self.n_proj)

        #     # max #
        #     regularization += tf.reduce_max(tf.math.square(tf.norm(loss_gradient, ord=2, axis=0)))
        # return regularization / (self.batch_size)

    def _projected_loss_regularizer(self, inputs, outputs):
        if DERIVATIVES_ON_INPUTS is False:
            raise AttributeError("\n You cannot compute partial derivatives w.r.t. projections in "
                                 "projected_loss regularizer. ")

        size = random.randint(MIN_SIZE, MAX_SIZE) if self.test is False else TEST_SIZE
        seed = random.randint(1, 100)
        print("\nsize =", size, ", seed =", seed)
        channels = self._get_n_channels(inputs)

        axis = 1 if self.data_format == "channels_first" else -1

        loss_gradient = 0
        for channel in range(channels):
            channel_data = tf.expand_dims(input=inputs[:,:,:,channel], axis=3)
            loss = K.categorical_crossentropy(target=outputs, output=self._get_logits(inputs=channel_data),
                                              from_logits=True, axis=axis)
            loss_gradient += self._compute_gradients(loss, [inputs])[0] / channels
        projected_loss = tf_flat_projection(input_data=loss_gradient, random_seed=seed, size_proj=size)[0]
        regularization = tf.reshape(projected_loss, shape=(self.batch_size, size, size, channels))
        regularization = tf.reduce_sum( tf.math.square(tf.norm(regularization, ord=2, axis=0)))
        return regularization / (self.batch_size * self.n_proj)

    def _loss_on_perturbations_regularizer(self, inputs, outputs):
        sess = tf.Session()
        sess.as_default()
        inputs = inputs.eval(session=sess)

        axis = 1 if self.data_format == "channels_first" else -1
        n_proj = random.randint(MIN_PROJ, MAX_PROJ)
        size = random.randint(MIN_SIZE, MAX_SIZE) if self.test is False else TEST_SIZE
        seeds = random.sample(range(1, 100), n_proj)
        print("\nsize =", size)

        projections, inverse_projections = compute_projections(inputs, seeds, n_proj, size, "channels")
        perturbations, augmented_inputs = compute_perturbations(inputs, inverse_projections)

        # === plot projections === #
        # print(projections[0,0,0,0,:],inverse_projections[0,0,0,0,:])
        # print(perturbations[0,0,0,:],augmented_inputs[0,0,0,:])
        # plot_projections([inputs,projections[0],inverse_projections[0],perturbations,augmented_inputs])
        # ======================== #

        loss = K.categorical_crossentropy(target=outputs, output=self._get_logits(inputs=augmented_inputs),
                                          from_logits=True, axis=axis)
        loss_gradient = self._compute_gradients(loss, [augmented_inputs])[0]
        loss_gradient = tf.cast(loss_gradient, tf.float32)

        regularization = tf.reduce_sum(tf.math.square(tf.norm(loss_gradient, ord=2, axis=0)))
        return regularization / self.batch_size

    def train(self, x_train, y_train):
        print("\nTraining infos:\nbatch_size = ", self.batch_size, "\nepochs = ", self.epochs,
              "\nx_train.shape = ", x_train.shape, "\ny_train.shape = ", y_train.shape, "\n")

        batches = int(len(x_train)/self.batch_size)
        x_train_batches = np.split(x_train, batches)
        y_train_batches = np.split(y_train, batches)

        start_time = time.time()
        for batch in range(batches):
            print("\n=== training batch", batch+1,"/",batches,"===")
            idxs = np.random.choice(len(x_train_batches[0]), self.batch_size, replace=False)
            x_train_sample = x_train_batches[batch][idxs]
            y_train_sample = y_train_batches[batch][idxs]
            inputs = tf.convert_to_tensor(x_train_sample)
            outputs = tf.convert_to_tensor(y_train_sample)
            early_stopping = keras.callbacks.EarlyStopping(monitor='loss', verbose=1)

            mini_batch = 20

            if self.projection_mode == "loss_on_perturbations":
                loss = self.loss_wrapper(inputs, outputs)
                self.model.compile(loss=loss, optimizer=keras.optimizers.Adadelta(lr=L_RATE), metrics=['accuracy'])
                self.model.fit(x_train_sample, y_train_sample, epochs=self.epochs, batch_size=mini_batch,
                               callbacks=[early_stopping])
            elif self.projection_mode == "no_projections":
                loss = self.loss_wrapper(inputs, outputs)
                self.model.compile(loss=loss, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
                self.model.fit(x_train_sample, y_train_sample, epochs=self.epochs, batch_size=mini_batch,
                               callbacks=[early_stopping])
            else:
                self.n_proj = random.randint(MIN_PROJ, MAX_PROJ) if self.test is False else TEST_PROJ
                print("\nn_proj =",self.n_proj)
                for proj in range(self.n_proj):
                    print("\nprojection",proj+1,"/",self.n_proj)
                    loss = self.loss_wrapper(inputs,outputs)
                    self.model.compile(loss=loss, optimizer=keras.optimizers.Adadelta(lr=L_RATE), metrics=['accuracy'])
                    self.model.fit(x_train_sample, y_train_sample, epochs=self.epochs, batch_size=mini_batch,
                                   callbacks=[early_stopping])

        print("\nTraining time: --- %s seconds ---" % (time.time() - start_time))
        return self

    def evaluate(self, classifier, x, y):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param classifier: trained classifier
        :param x: test data
        :param y: test labels
        :return: predictions
        """
        # setting classes_ attribute for loaded models
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)

        # return self.adversarial_classifier.evaluate(classifier, x, y)
        # evaluate the performance on the adversarial test set
        y_pred = classifier.predict(x)
        y_true = np.argmax(y, axis=1)
        nb_correct_adv_pred = np.sum(y_pred == y_true)

        print("\nAdversarial test data.")
        print("Correctly classified: {}".format(nb_correct_adv_pred))
        print("Incorrectly classified: {}".format(len(x) - nb_correct_adv_pred))

        acc = nb_correct_adv_pred / y.shape[0]
        print("Adversarial accuracy: %.2f%%" % (acc * 100))

        # classification report
        print(classification_report(np.argmax(y, axis=1), y_pred, labels=range(self.num_classes)))

    # todo: deprecated
    def evaluate_test(self, x_test, y_test):
        """
        Evaluates the trained classifier on the given test set and computes the accuracy on the predictions.
        :param classifier: trained classifier
        :param x_test: test data
        :param y_test: test labels
        :return: x_test predictions
        """

        # setting classes_ attribute for loaded models
        if len(y_test.shape) == 2 and y_test.shape[1] > 1:
            self.classes_ = np.arange(y_test.shape[1])
        elif (len(y_test.shape) == 2 and y_test.shape[1] == 1) or len(y_test.shape) == 1:
            self.classes_ = np.unique(y_test)

        print("\n===== Test set evaluation =====")
        print("\nTesting infos:\nx_test.shape = ", x_test.shape, "\ny_test.shape = ", y_test.shape, "\n")

        y_test_pred = self.predict(x_test)
        y_test_true = np.argmax(y_test, axis=1)
        correct_preds = np.sum(y_test_pred == np.argmax(y_test, axis=1))

        print("Correctly classified: {}".format(correct_preds))
        print("Incorrectly classified: {}".format(len(x_test) - correct_preds))

        acc = np.sum(y_test_pred == y_test_true) / y_test.shape[0]
        print("Test accuracy: %.2f%%" % (acc * 100))

        # classification report over single classes
        print(classification_report(np.argmax(y_test, axis=1), y_test_pred, labels=range(self.num_classes)))

        return y_test_pred

    # todo: deprecated
    def evaluate_adversaries(self, x_test, y_test, method, dataset_name, adversaries_path=None, test=False):
        print("\n===== Adversarial evaluation =====")

        if len(y_test.shape) == 2 and y_test.shape[1] > 1:
            self.classes_ = np.arange(y_test.shape[1])
        elif (len(y_test.shape) == 2 and y_test.shape[1] == 1) or len(y_test.shape) == 1:
            self.classes_ = np.unique(y_test)

        # generate adversaries on the test set
        x_test_adv = self._get_adversaries(self, x_test, y_test, method=method, dataset_name=dataset_name,
                                           adversaries_path=adversaries_path, test=test)

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

    def load_adversaries(self, dataset_name, attack, eps, test):
        return self.adversarial_classifier.load_adversaries(dataset_name, attack, eps, test)

    # todo: implement generate adversaries using method from adversarial classifier
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
            if dataset_name == "mnist":
                # todo: buggy mnist test data
                # mnist x_test data was saved incorrectly together with prediction labels y_test, so I'm only taking
                # the first element in the list.
                x_adv = load_from_pickle(path=adversaries_path, test=test)[0]
            else:
                x_adv = load_from_pickle(path=adversaries_path, test=test)


        return x_adv

    def save_classifier(self, classifier):
        """
        Saves the trained model and adds the current datetime to the filepath.

        :param classifier: trained classifier
        """
        modelname = str(dataset_name) + "_randreg_lam=" + str(classifier.lam) + \
                    "_batch=" + str(classifier.batch_size) + "_epochs=" + str(classifier.epochs) + \
                    "_" + str(classifier.projection_mode) + ".h5"
        model_path = RESULTS + time.strftime('%Y-%m-%d') + "/" + modelname
        classifier.model.save_weights(model_path)

    def load_classifier(self, current_day_folder=False):
        """
        Loads a pretrained classifier. It load either the baseline model or the adversarially trained robust version.
        :param dataset_name: dataset name
        :param attack: attack method for loading adversarially trained robust models
        :param eps: threshold for the norm of a perturbation
        returns: trained classifier
        """
        modelname = str(dataset_name) + "_randreg_lam=" + str(self.lam) + "_epochs=" + str(self.epochs) + \
                    "_" + str(self.projection_mode) + ".h5"
        if current_day_folder:
            model_path = RESULTS + time.strftime('%Y-%m-%d') + "/" + modelname
        else:
            model_path = TRAINED_MODELS + modelname

        self.model.load_weights(model_path)
        return self

def main(dataset_name, test, lam, projection_mode, eps):

    # === Tensorflow session and initialization === #
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    os.makedirs(os.path.dirname(RESULTS+time.strftime('%Y-%m-%d')+"/"), exist_ok=True)

    # === model initialization === #
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name, test=test)

    randreg = RandomRegularizer(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                   dataset_name=dataset_name, sess=sess, lam=lam, projection_mode=projection_mode,
                                   test=test,eps=eps)

    # === train === #
    # classifier = randreg.train(x_train, y_train)
    # randreg.save_classifier(classifier)
    classifier = randreg.load_classifier(current_day_folder=False)

    # === evaluate === #
    randreg.evaluate(classifier=classifier, x=x_test, y=y_test)

    ######### todo: bug on adversarial evaluation. It only works on loaded models..
    # del randreg, classifier
    # randreg = RandomRegularizer(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
    #                                dataset_name=dataset_name, sess=sess, lam=lam, projection_mode=projection_mode,
    #                                test=test,eps=eps)
    # classifier = randreg.load_classifier(current_day_folder=True)
    #########

    # === baseline adversaries === #
    for method in ['fgsm','pgd','deepfool','carlini_linf']:
        x_test_adv = randreg.load_adversaries(dataset_name, method, eps, test)
        randreg.evaluate(classifier, x_test_adv, y_test)

    # todo: refactor
    # print("\nRandreg adversaries")
    # for attack in ['fgsm','pgd','deepfool','carlini_linf']:
    #     filename = dataset_name + "_x_test_" + attack + "_randreg.pkl"
    #     x_test_adv = classifier.evaluate_adversaries(x_test, y_test, method=attack, dataset_name=dataset_name,
    #                                                  #adversaries_path=DATA_PATH+filename,
    #                                                  test=test)
    #     # save_to_pickle(data=x_test_adv, filename=filename)


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        lam = float(sys.argv[3])
        projection_mode = sys.argv[4]
        eps = float(sys.argv[5])

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        lam = float(input("\nChoose lambda regularization weight (type=float): "))
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")
        eps = float(input("\nSet a ths for perturbation norm: "))

    main(dataset_name=dataset_name, test=test, lam=lam, projection_mode=projection_mode, eps=eps)

