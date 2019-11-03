# -*- coding: utf-8 -*-

from tensorflow import keras
import keras.losses
from random_ensemble import *
from projection_functions import *
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

############
# defaults #
############

MIN_SIZE = 10
MAX_SIZE = 20
MIN_PROJ = 2
MAX_PROJ = 4

L_RATE = 5
TEST_SIZE = 2
TEST_PROJ = 1
PROJ_MODE = "no_projections, loss_on_projections, projected_loss, loss_on_perturbations"
CHANNEL_MODE = "channels"  # "channels, grayscale"
DERIVATIVES_ON_INPUTS = True  # if True compute gradient derivatives w.r.t. the inputs, else w.r.t. the projected inputs
TRAINED_MODELS = "../trained_models/random_regularizer/"  # RESULTS + time.strftime('%Y-%m-%d') + "/"


class RandomRegularizer(BaselineConvnet):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, lam, projection_mode, test, init_seed=0):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        super(RandomRegularizer, self).__init__(input_shape, num_classes, data_format, dataset_name, test)
        self.init_seed = init_seed
        self.lam = lam
        self.n_proj = None
        self.projection_mode = projection_mode
        self.inputs = Input(shape=self.input_shape)
        self.model_name = str(dataset_name) + "_randreg_lam=" + str(self.lam) + \
                          "_" + str(self.projection_mode) + "_" + str(self.init_seed) + ".h5"

        print("\nprojection mode =", self.projection_mode, ", channel mode =", CHANNEL_MODE,", lambda =", self.lam,
              ", init_seed = ", self.init_seed)
        print("\nbatch_size =",self.batch_size,", epochs =",self.epochs,", lr =",L_RATE)
        print("\nn_proj~(",MIN_PROJ,",",MAX_PROJ,"), size_proj~(",MIN_SIZE,",",MAX_SIZE,")")

    def _get_proj_params(self):
        random.seed(self.init_seed)
        size = random.randint(MIN_SIZE, MAX_SIZE) if self.test is False else TEST_SIZE
        seed = random.randint(1, 100)  # random.sample(range(1, 100), n_proj) # list of seeds
        return {"size":size,"seed":seed}

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

    def _get_n_channels(self, inputs):
        return inputs.get_shape().as_list()[3]

    def _no_projections_regularizer(self, inputs, outputs):
        if DERIVATIVES_ON_INPUTS is False:
            raise AttributeError("\nYou can only compute derivatives on the inputs. Set DERIVATIVES_ON_INPUTS = True.")

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

        channels = self._get_n_channels(inputs)
        size, seed = self._get_proj_params().values()

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

        size, seed = self._get_proj_params().values()
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
        regularization = tf.reduce_sum(tf.math.square(tf.norm(regularization, ord=2, axis=0)))
        return regularization / (self.batch_size * self.n_proj)

    def _loss_on_perturbations_regularizer(self, inputs, outputs):
        sess = tf.Session()
        sess.as_default()
        inputs = inputs.eval(session=sess)

        axis = 1 if self.data_format == "channels_first" else -1
        n_proj = random.randint(MIN_PROJ, MAX_PROJ)
        size, _ = self._get_proj_params().values()
        seeds = random.sample(range(1, 100), n_proj)

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

    def train(self, x_train, y_train, device):
        print("\nTraining infos:\nbatch_size = ", self.batch_size, "\nepochs = ", self.epochs,
              "\nx_train.shape = ", x_train.shape, "\ny_train.shape = ", y_train.shape, "\n")

        # print(len(x_train), self.batch_size)
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

            mini_batch = MINIBATCH
            device_name = self._set_device_name(device)
            with tf.device(device_name):
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


def main(dataset_name, test, lam, projection_mode, eps, device, seed):
    """
    :param dataset_name: choose between "mnist" and "cifar"
    :param test: if True only takes the first 100 samples
    :param lam: lambda regularization weight parameter
    :param projection_mode: method for computing projections on RGB images
    :param eps: upper ths for adversaries distance
    :param device: code execution device (cpu/gpu)
    :param seed: random seed for the projections
    """

    # === initialize === #
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name, test=test)

    randreg = RandomRegularizer(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                dataset_name=dataset_name, lam=lam, projection_mode=projection_mode, test=test,
                                init_seed=seed)

    # === train === #
    randreg.train(x_train, y_train, device)
    # randreg.save_classifier()
    # randreg.load_classifier(relative_path=RESULTS + time.strftime('%Y-%m-%d') + "/")
    # randreg.load_classifier(relative_path=TRAINED_MODELS)

    # === evaluate === #
    randreg.evaluate(x=x_test, y=y_test)
    for method in ['fgsm','pgd','deepfool','carlini']:
        x_test_adv = randreg.load_adversaries(dataset_name, method, eps, test)
        randreg.evaluate(x_test_adv, y_test)


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        lam = float(sys.argv[3])
        projection_mode = sys.argv[4]
        eps = float(sys.argv[5])
        device = sys.argv[6]
        seed = int(sys.argv[7])

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        lam = float(input("\nChoose lambda regularization weight (type=float): "))
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")
        eps = float(input("\nSet a ths for perturbation norm: "))
        device = input("\nChoose a device (cpu/gpu): ")
        seed = input("\nSet a random seed (int>=0).")

    main(dataset_name=dataset_name, test=test, lam=lam, projection_mode=projection_mode, eps=eps, device=device,
         seed=seed)

