import numpy as np
import keras
from keras import backend as K
from keras.datasets import mnist
import pickle as pkl
import time
from sklearn.random_projection import GaussianRandomProjection
import os
import matplotlib.pyplot as plt
import tensorflow as tf

MIN = 0
MAX = 255
TEST_SIZE = 100
RESULTS = "../results/"

######################
# data preprocessing #
######################


def preprocess_mnist(test, img_rows=28, img_cols=28):
    """Preprocess mnist dataset for keras training

    :param test: If test is True, only load the first 100 images
    :param img_rows: input image n. rows
    :param img_cols: input image n. cols
    """
    print("\nLoading mnist.")

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape, '\nx_test shape:', x_test.shape,)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if test:
        x_train = x_train[:TEST_SIZE]
        y_train = y_train[:TEST_SIZE]
        x_test = x_test[:TEST_SIZE]
        y_test = y_test[:TEST_SIZE]

    num_classes = 10
    data_format = 'channels_last'
    return x_train, y_train, x_test, y_test, input_shape, num_classes, data_format


def _onehot(integer_labels):
    """Return matrix whose rows are onehot encodings of integers."""
    n_rows = len(integer_labels)
    n_cols = integer_labels.max() + 1
    onehot = np.zeros((n_rows, n_cols), dtype='uint8')
    onehot[np.arange(n_rows), integer_labels] = 1
    return onehot


def load_cifar(test):
    """Return train_data, train_labels, test_data, test_labels
    The shape of data is 32 x 32 x3"""
    x_train = None
    y_train = []

    data_dir='../data/cifar-10/'

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "data_batch_{}".format(i))
        if i == 1:
            x_train = data_dic['data']
        else:
            x_train = np.vstack((x_train, data_dic['data']))
        y_train += data_dic['labels']

    test_data_dic = unpickle(data_dir + "test_batch")
    x_test = test_data_dic['data']
    y_test = test_data_dic['labels']

    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    x_train = np.rollaxis(x_train, 1, 4)
    y_train = np.array(y_train)

    x_test = x_test.reshape((len(x_test), 3, 32, 32))
    x_test = np.rollaxis(x_test, 1, 4)
    y_test = np.array(y_test)

    input_shape = x_train.shape[1:]
    num_classes = 10
    data_format = 'channels_first'

    if test:
        x_train = x_train[:TEST_SIZE]
        y_train = y_train[:TEST_SIZE]
        x_test = x_test[:TEST_SIZE]
        y_test = y_test[:TEST_SIZE]

    return x_train, _onehot(y_train), x_test, _onehot(y_test), input_shape, num_classes, data_format


def load_dataset(dataset_name, test):
    """
    Load dataset.
    :param dataset_name: choose between "mnist" and "cifar"
    :param test: If True only loads the first 100 samples
    """
    global x_train, y_train, x_test, y_test, input_shape, num_classes, data_format

    if dataset_name == "mnist":
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = preprocess_mnist(test=test)
    elif dataset_name == "cifar":
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_cifar(test=test)

    return x_train, y_train, x_test, y_test, input_shape, num_classes, data_format

######################
# random projections #
######################


# todo: remove, deprecated
# def one_channel_projection_old(input_data, random_seed, size_proj, channel=0):
#     """ Computes one projection over a single channel of the whole input data.
#      It samples `size_proj` random directions for the projection using the given random_seed.
#
#      :param input_data: original input data
#      :param random_seed: projection seed
#      :param size_proj: size of a projection
#      :return: np.array containing the projected version of input_data
#      """
#     samples, rows, cols, channels = input_data.shape
#
#     projector = GaussianRandomProjection(n_components=size_proj*size_proj, random_state=random_seed)
#
#     flat_images = input_data.reshape((samples, rows*cols, channels))
#     projection = np.empty((samples, size_proj, size_proj, channels))
#
#     for im in range(samples):
#         projected_image = np.empty(shape=(size_proj * size_proj, channels))
#         projected_image[:, channel] = projector.fit_transform(flat_images[im, :, channel].reshape(1, -1))
#         projection[im, :, :] = projected_image.reshape((size_proj, size_proj, channels))
#
#     return projection


# todo: this is only used in parallel implementation, extend it to the other methods.
def compute_single_projection_par(input_data, random_seed, size_proj):
    """ Computes one projection of the whole input data over `size_proj` randomly chosen directions with Gaussian
         matrix entries sampling, using the given random_seed.

     :param input_data: high dimensional input data
     :param random_seed: projection seed
     :param size_proj: size of a projection
     :return: np.array containing all random projections of input_data
     """
    projector = GaussianRandomProjection(n_components=size_proj*size_proj, random_state=random_seed)
    flat_images = input_data.reshape((input_data.shape[0], input_data.shape[1] * input_data.shape[2], input_data.shape[3]))
    single_projection = np.empty((input_data.shape[0], size_proj, size_proj, input_data.shape[3]))

    #channel_projection = np.empty(shape=(input_data.shape[0], size_proj * size_proj))
    for channel in range(input_data.shape[3]):
        channel_projection = projector.fit_transform(flat_images[:, :, channel]) \
            .reshape((input_data.shape[0], size_proj, size_proj))
        single_projection[:, :, :, channel] = channel_projection

    return single_projection


def flat_projection(input_data, random_seed, size_proj):
    """ Computes a projection of the whole input data flattened over channels and also computes the inverse projection.
    It samples `size_proj` random directions for the projection using the given random_seed.

    :param input_data: high dimensional input data
    :param random_seed: projection seed
    :param size_proj: size of a projection
    :return:
    :param projection: np.array containing a random projection of input_data
    :param projection: np.array containing the inverse projection of input_data given by the
    pseudoinverse of the projection matrix
    """
    # this is needed to go from tf tensors to np arrays:
    sess = tf.Session()
    sess.as_default()
    samples, rows, cols, channels = input_data.shape

    # projection matrices
    projector = GaussianRandomProjection(n_components=size_proj * size_proj * channels, random_state=random_seed)
    proj_matrix = np.float32(projector._make_random_matrix(size_proj * size_proj * channels, rows * cols * channels))
    pinv = np.linalg.pinv(proj_matrix)

    # compute projections
    flat_images = tf.cast(input_data.reshape((samples, rows*cols*channels)), dtype=tf.float32)
    projection = tf.matmul(a=flat_images, b=proj_matrix, transpose_b=True)
    inverse_projection = tf.matmul(a=projection, b=pinv, transpose_b=True)

    # reshape
    projection = tf.reshape(projection, shape=(samples, size_proj, size_proj, channels)).eval(session=sess)
    inverse_projection = tf.reshape(inverse_projection, shape=(input_data.shape)).eval(session=sess)
    return projection, inverse_projection


def channels_projection(input_data, random_seed, size_proj):
    """ Computes a projection of the whole input data over each channel, then reconstructs the rgb image.
    It also computes the inverse projections.
    It samples `size_proj` random directions for the projection using the given random_seed.

    :param input_data: high dimensional input data
    :param random_seed: projection seed
    :param size_proj: size of a projection
    :return:
    :param projection: np.array containing a random projection of input_data
    :param projection: np.array containing the inverse projection of input_data given by the
    pseudoinverse of the projection matrix
    """

    sess = tf.Session()
    sess.as_default()
    samples, rows, cols, channels = input_data.shape

    # projection_matrices
    projector = GaussianRandomProjection(n_components=size_proj * size_proj, random_state=random_seed)
    proj_matrix = np.float32(projector._make_random_matrix(size_proj * size_proj, rows * cols))
    pinv = np.linalg.pinv(proj_matrix)

    # compute projections
    flat_images = tf.cast(input_data.reshape((samples, rows * cols, channels)), dtype=tf.float32)
    projection = np.empty((samples, size_proj, size_proj, channels))
    inverse_projection = np.empty(input_data.shape)
    for channel in range(channels):
        # projection
        channel_projections = tf.matmul(a=flat_images[:, :, channel], b=proj_matrix, transpose_b=True)
        im_channel_projections = tf.reshape(channel_projections, shape=(samples, size_proj, size_proj))
        projection[:, :, :, channel] = im_channel_projections.eval(session=sess)
        # inverse projection
        channel_inverse_projections = tf.matmul(a=channel_projections, b=pinv, transpose_b=True)
        im_channel_inverse_projections = tf.reshape(channel_inverse_projections, shape=(samples, rows, cols))
        inverse_projection[:, :, :, channel] = im_channel_inverse_projections.eval(session=sess)

    return projection, inverse_projection


def one_channel_projection(input_data, random_seed, size_proj, channel=0):
    samples, rows, cols, channels = input_data.shape
    single_channel = input_data[:,:,:,channel].reshape(samples, rows, cols, 1)
    return flat_projection(single_channel, random_seed, size_proj)


def grayscale_projection(input_data, random_seed, size_proj):
    samples, rows, cols, channels = input_data.shape
    greyscale_data = np.array([rgb2gray(rgb_im) for rgb_im in input_data]).reshape((samples, rows, cols, 1))
    return flat_projection(greyscale_data, random_seed, size_proj)

def compute_single_projection(input_data, seed, size_proj, projection_mode):
    # print("\ncomputing",projection_mode,"projection")

    projection = None
    inverse_projection = None
    if projection_mode == "flat":
        projection, inverse_projection = flat_projection(input_data=input_data, size_proj=size_proj,
                                                         random_seed=seed)
    elif projection_mode == "channels":
        projection, inverse_projection = channels_projection(input_data=input_data, size_proj=size_proj,
                                                             random_seed=seed)
    elif projection_mode == "one_channel":
        projection, inverse_projection = one_channel_projection(input_data=input_data, size_proj=size_proj,
                                                                random_seed=seed)
    elif projection_mode == "grayscale":
        projection, inverse_projection = grayscale_projection(input_data=input_data, size_proj=size_proj,
                                                              random_seed=seed)
    return projection,inverse_projection

def compute_projections(input_data, random_seeds, n_proj, size_proj, projection_mode):
    """ Computes `n_proj` projections of the whole input data over `size_proj` randomly chosen directions, using a
    given list of random seeds.

    :param input_data: high dimensional input data
    :param random_seeds: list of random seeds for the projections
    :param n_proj: number of projections
    :param size_proj: size of a projection
    :return: np.array containing n_proj random projections on the data
    """
    # this is needed to go from tf tensors to np arrays:
    sess = tf.Session()
    sess.as_default()

    print("Input shape: ", input_data.shape)
    print("\nComputing ",n_proj,"random projections in ",projection_mode,"mode: ")
    # samples, rows, cols, channels = input_data.shape
    # projections = np.empty(shape=(n_proj, samples, size_proj, size_proj, channels))
    # inverse_projections = np.empty(shape=(n_proj, samples, rows, cols, channels))

    projections = []
    inverse_projections = []
    for proj_idx in range(n_proj):
        # all_projections[proj_idx,:,:,:,:] = compute_single_projection_channel(input_data=input_data,
        #                                     size_proj=size_proj, random_seed=random_seeds[proj_idx])

        projection, inverse_projection = compute_single_projection(input_data, random_seeds[proj_idx], size_proj, projection_mode)
        projections.append(projection)
        inverse_projections.append(inverse_projection)

    projections = np.array(projections)
    inverse_projections = np.array(inverse_projections)
    print("Projected data dimensions:", projections.shape)

    return projections, inverse_projections

################
# Pickle utils #
################


def save_to_pickle(data, filename):
    """ saves data to pickle """
    os.makedirs(os.path.dirname(RESULTS+time.strftime('%Y-%m-%d')+"/"+filename), exist_ok=True)
    with open(RESULTS+time.strftime('%Y-%m-%d')+"/"+filename, 'wb') as f:
        pkl.dump(data, f)


def unpickle(file):
    """ Load byte data from file"""
    with open(file, 'rb') as f:
        data = pkl.load(f, encoding='latin-1')
    return data


def load_from_pickle(path, test):
    """ loads data from pickle containing: x_test, y_test."""
    with open(path, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        # In the pickles I'm also saving the labels, so here I only take the data.
        # todo: solve this issue by adding a special case for the old pkl data or simply transform it.
        # when loading only x_test it should become:
        # data = u.load()
        data = u.load()[0]
    if test is True:
        data = data[:TEST_SIZE]
    return data

##############
# plot utils #
##############


def plot_projected_images(input_data, projected_data, n_proj, im_idxs):
    # todo. deprecated, remove this
    fig, axs = plt.subplots(len(im_idxs),n_proj+1,figsize=(10, 8))
    for im_idx, im in enumerate(im_idxs):
        axs[im_idx, 0].imshow(np.squeeze(input_data)[im_idx])
        for proj in range(n_proj):
            axs[im_idx, proj+1].imshow(np.squeeze(projected_data)[proj][im])

    # block plots at the end of code execution
    plt.show(block=False)
    input("Press ENTER to exit")
    exit()


def plot_inverse_projections(input_data, random_seeds, n_proj, size_proj, projection_mode, test=False, im_idxs=range(10), proj_idx=0):
    """
    Plots input data in the first row, projected data in the second row and inverse projected data in the third row.
    By default it only takes the first 10 samples and the first projection.

    :param input_data: high dimensional input data
    :param random_seeds: list of random seeds for the projections
    :param n_proj: number of projections
    :param size_proj: size of a projection
    """

    projections, inverse_projections = compute_projections(input_data, random_seeds, n_proj, size_proj, projection_mode)
    # print(input_data.shape, projections.shape, inverse_projections.shape)

    fig, axs = plt.subplots(nrows=3, ncols=len(im_idxs), figsize=(10, 8))

    # cmap = "gray" if input_data.shape[3] == 1 else None
    cmap = None
    for im_idx in range(len(im_idxs)):
        axs[0, im_idx].imshow(np.squeeze(input_data[im_idx]), cmap=cmap)
        axs[1, im_idx].imshow(np.squeeze(projections[proj_idx,im_idx]), cmap=cmap)
        axs[2, im_idx].imshow(np.squeeze(inverse_projections[proj_idx,im_idx]), cmap=cmap)

    if test == False:
        # If not in testing mode, block imshow.
        plt.show(block=False)
        input("Press ENTER to exit")
        exit()


def rgb2gray(rgb):
    """ convert rgb image to greyscale image """
    if rgb.shape[2] == 1:
        return rgb
    else:
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
