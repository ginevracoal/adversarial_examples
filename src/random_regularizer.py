# -*- coding: utf-8 -*-

from tensorflow import keras
import keras.losses
from random_ensemble import *
from projection_functions import *
import tensorflow as tf
import random
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

############
# defaults #
############

MIN_SIZE = 10
MAX_SIZE = 25
MIN_PROJ = 1
MAX_PROJ = 1

L_RATE = 0.01
TEST_SIZE = 2
TEST_PROJ = 1
PROJ_MODE = "no_projections, loss_on_projections, projected_loss, loss_on_perturbations"
CHANNEL_MODE = "channels"  # "channels, grayscale"
DERIVATIVES_ON_INPUTS = True  # if True compute gradient derivatives w.r.t. the inputs, else w.r.t. the projected inputs
TRAINED_MODELS = "../trained_models/"
MODEL_NAME = "randreg"


class RandomRegularizer(BaselineConvnet):

    def __init__(self, input_shape, num_classes, data_format, dataset_name, lam, projection_mode, test, library,
                 seed=0, epochs=None):
        """
        :param dataset_name: name of the dataset is required for setting different CNN architectures.
        """
        self.seed = seed
        self.projection_mode = projection_mode
        super(RandomRegularizer, self).__init__(input_shape, num_classes, data_format, dataset_name, test, library,
                                                epochs)
        self.lam = lam
        self.n_proj = 3 # projections per batch
        self.batch_size = 1000
        self.inputs = Input(shape=self.input_shape)
        # self.model_name = str(dataset_name) + "_randreg_lam=" + str(self.lam) + \
        #                   "_" + str(self.projection_mode) + "_" + str(self.init_seed)

        print("\nprojection mode =", self.projection_mode, ", channel mode =", CHANNEL_MODE,", lambda =", self.lam,
              ", init_seed = ", self.seed)
        print("\nbatch_size =",self.batch_size,", epochs =",self.epochs,", lr =",L_RATE)
        print("\nsize_proj~(",MIN_SIZE,",",MAX_SIZE,")", ", proj_per_batch=", self.n_proj)

    def _get_proj_params(self):
        random.seed(self.seed)
        size = random.randint(MIN_SIZE, MAX_SIZE) if self.test is False else TEST_SIZE
        seed = random.randint(1, 100)  # random.sample(range(1, 100), n_proj) # list of seeds
        return {"size":size,"seed":seed}

    def _set_model_path(self, model_name="randreg"):
        """ Defines model path and filename """
        folder = str(model_name)+"/"
        if self.epochs == None:
            filename = self.dataset_name + "_" + str(model_name) + "_" + str(self.projection_mode) + "_seed=" \
                       + str(self.seed)
        else:
            filename = self.dataset_name + "_" + str(model_name) + "_epochs=" + str(self.epochs) + "_" \
                       + str(self.projection_mode) +"_seed="+str(self.seed)
        return {'folder': folder, 'filename': filename}

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

        self.loss = custom_loss
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
                                                                                random_seed=seed, size_proj=size,
                                                                                translation=None)[1]),
                                                  from_logits=True, axis=axis)
                loss_gradient = self._compute_gradients(loss, [channel_inputs])[0]
            else:
                pinv_channel_inputs = tf_flat_projection(input_data=channel_inputs, random_seed=seed, size_proj=size)[1]
                loss = K.categorical_crossentropy(target=outputs, output=self._get_logits(inputs=pinv_channel_inputs),
                                                  from_logits=True, axis=axis)
                loss_gradient = self._compute_gradients(loss, [pinv_channel_inputs])[0]

            # expectation #
            regularization += tf.reduce_sum(tf.math.square(tf.norm(loss_gradient/channels, ord=2, axis=0)))
        return regularization / self.batch_size #(self.batch_size * self.n_proj)

        #     # max #
        #     regularization += tf.reduce_max(tf.math.square(tf.norm(loss_gradient, ord=2, axis=0)))
        # return regularization / (self.batch_size)

    def _projected_loss_regularizer(self, inputs, outputs):
        if DERIVATIVES_ON_INPUTS is False:
            raise AttributeError("\n You cannot compute partial derivatives w.r.t. projections in "
                                 "projected_loss regularizer. ")

        channels = self._get_n_channels(inputs)
        axis = 1 if self.data_format == "channels_first" else -1

        # compute loss gradients
        loss_gradient = 0
        for channel in range(channels):
            channel_data = tf.expand_dims(input=inputs[:,:,:,channel], axis=3)
            loss = K.categorical_crossentropy(target=outputs, output=self._get_logits(inputs=channel_data),
                                              from_logits=True, axis=axis)
            loss_gradient += self._compute_gradients(loss, [inputs])[0] / channels

        # project and regularize
        size = self._get_proj_params()['size']
        regularization = 0
        for proj in range(self.n_proj):
            seed = self._get_proj_params()['seed']
            projected_loss = tf_flat_projection(input_data=loss_gradient, random_seed=seed, size_proj=size)[0]
            # regularization = tf.reshape(projected_loss, shape=(self.batch_size, size, size, channels))
            regularization += tf.reduce_sum(tf.math.square(tf.norm(projected_loss, ord=2, axis=0))) / self.n_proj
        return regularization / self.batch_size

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
        """
        Separates the training data into batches, computes a random projection of each batch
        :param x_train: input samples
        :param y_train: input labels
        :param device: device for computations (cpu/gpu)
        :return: trained model (self)
        """
        print("\nTraining infos:\nbatch_size = ", self.batch_size, "\nepochs = ", self.epochs,
              "\nx_train.shape = ", x_train.shape, "\ny_train.shape = ", y_train.shape, "\n")

        # batches = int(len(x_train)/self.batch_size)
        n_batches = 1 if self.test else int(len(x_train)/self.batch_size)
        print("n_batches = ", n_batches)
        # self.batch_size = int(len(x_train)/n_batches)
        # print(len(x_train), self.batch_size, n_batches)
        x_train_batches = np.split(x_train, n_batches)
        y_train_batches = np.split(y_train, n_batches)
        start_time = time.time()

        for batch in range(n_batches):
            print("\n=== training batch", batch+1,"/",n_batches,"===")
            # idxs = np.random.choice(len(x_train_batches[0]), self.batch_size, replace=False)
            # x_train_sample = x_train_batches[batch][idxs]
            # y_train_sample = y_train_batches[batch][idxs]
            # inputs = tf.convert_to_tensor(x_train_sample)
            # outputs = tf.convert_to_tensor(y_train_sample)
            x_train_batch = x_train_batches[batch]
            y_train_batch = y_train_batches[batch]
            inputs = tf.convert_to_tensor(x_train_batch)
            outputs = tf.convert_to_tensor(y_train_batch)

            mini_batch = MINIBATCH
            device_name = self._set_device_name(device)
            with tf.device(device_name):
                loss = self.loss_wrapper(inputs,outputs)
                self.model.compile(loss=loss, optimizer=keras.optimizers.Adadelta(lr=L_RATE), metrics=['accuracy'])
                if self.epochs == None:
                    es = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1)
                    self.model.fit(x_train_batch, y_train_batch, epochs=80, batch_size=mini_batch,
                                   callbacks=[es], shuffle=True, validation_split=0.2)
                else:
                    self.model.fit(x_train_batch, y_train_batch, epochs=self.epochs, batch_size=mini_batch,
                                   shuffle=True, validation_split=0.2)

            # # intermediate evaluations
            # x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(
            #     dataset_name=dataset_name,
            #     test=test)
            # self.evaluate(x=x_test, y=y_test)
            # for method in ['fgsm', 'pgd', 'deepfool', 'carlini', 'newtonfool']:
            #     x_test_adv = self.load_adversaries(attack=method, seed=seed)
            #     self.evaluate(x_test_adv, y_test)

        print("\nTraining time: --- %s seconds ---" % (time.time() - start_time))
        self.trained = True
        return self

    def load_classifier(self, relative_path, folder=None, filename=None):
        if folder is None:
            folder = self.folder
        if filename is None:
            filename = self.filename
        print("\nLoading model: ", relative_path + folder + filename + ".h5")
        self.model = load_model(relative_path + folder + filename + ".h5", custom_objects={'custom_loss':self.loss})
        self.trained = True
        return self

def main(dataset_name, test, lam, projection_mode, device, seed):
    """
    :param dataset_name: choose between "mnist" and "cifar"
    :param test: if True only takes the first 100 samples
    # :param lam: lambda regularization weight parameter
    # :param projection_mode: method for computing projections on RGB images
    # :param eps: upper ths for adversaries distance
    :param device: code execution device (cpu/gpu)
    :param seed: random seed for the projections
    """


    # === initialize === #
    random.seed(seed)
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name=dataset_name,
                                                                                           test=test)

    randreg = RandomRegularizer(input_shape=input_shape, num_classes=num_classes, data_format=data_format,
                                dataset_name=dataset_name, lam=lam, projection_mode=projection_mode, test=test,
                                seed=seed, library="cleverhans")

    # === train === #
    randreg.train(x_train, y_train, device)
    randreg.save_classifier(relative_path=RESULTS)#, filename=randreg.filename+"_seed="+str(seed))
    # randreg.load_classifier(relative_path=RESULTS + time.strftime('%Y-%m-%d') + "/")
    # randreg.load_classifier(relative_path=RESULTS)

    # === evaluate === #
    randreg.evaluate(x=x_test, y=y_test)
    for method in ['fgsm','pgd','deepfool','virtual', 'spatial']:
        x_test_adv = randreg.load_adversaries(attack=method, seed=0, relative_path=DATA_PATH)
        randreg.evaluate(x_test_adv, y_test)


if __name__ == "__main__":
    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        lam = float(sys.argv[3])
        projection_mode = sys.argv[4]
        device = sys.argv[5]
        seed = int(sys.argv[6])

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        lam = float(input("\nChoose lambda regularization weight (type=float): "))
        projection_mode = input("\nChoose projection mode ("+PROJ_MODE+"): ")
        device = input("\nChoose a device (cpu/gpu): ")
        seed = input("\nSet a random seed (int>=0).")

    main(dataset_name=dataset_name, test=test, lam=lam, projection_mode=projection_mode, device=device, seed=seed)

