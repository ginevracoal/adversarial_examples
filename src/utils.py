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
    # todo: docstring types

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
    # todo: docstring types
    """

    sess = tf.Session()
    sess.as_default()
    samples, rows, cols, channels = input_data.shape
    projection = np.empty((samples, size_proj, size_proj, channels))
    inverse_projection = np.empty(input_data.shape)
    for channel in range(channels):
        single_channel = input_data[:, :, :, channel].reshape(samples, rows, cols, 1)
        channel_projection, channel_inverse_projection = flat_projection(single_channel, random_seed, size_proj)
        projection[:, :, :, channel] = np.squeeze(channel_projection)
        inverse_projection[:, :, :, channel] = np.squeeze(channel_inverse_projection)

    return projection, inverse_projection


def grayscale_projection(input_data, random_seed, size_proj):
    samples, rows, cols, channels = input_data.shape
    greyscale_data = np.array([rgb2gray(rgb_im) for rgb_im in input_data]).reshape((samples, rows, cols, 1))
    return flat_projection(greyscale_data, random_seed, size_proj)


def compute_single_projection(input_data, seed, size_proj, projection_mode):
    """
    Computes a single projection of the whole input data and the associated inverse projection using Moore-Penrose
     pseudoinverse.
    :param input_data: high dimensional input data, type=np.ndarray, shape=(batch_size, rows, cols, channels)
    :param seed: random seed for projection
    :param size_proj: size for the projections
    :param projection_mode: choose computation mode for projections.
                            Supported modes are "flat","channels", "one_channel" and "grayscale"
    :return:
    :param projection: projection of the whole input_data
                       type=np.ndarray, shape=(batch_size, size, size, channels)
    :param inverse_projection: inverse projection of the projected points
                               type=np.ndarray, shape=(batch_size, rows, cols, channels)
    """

    projection = None
    inverse_projection = None
    if projection_mode == "flat":
        projection, inverse_projection = flat_projection(input_data=input_data, size_proj=size_proj,
                                                         random_seed=seed)
    elif projection_mode == "channels":
        projection, inverse_projection = channels_projection(input_data=input_data, size_proj=size_proj,
                                                             random_seed=seed)
    elif projection_mode == "grayscale":
        projection, inverse_projection = grayscale_projection(input_data=input_data, size_proj=size_proj,
                                                              random_seed=seed)
    return projection, inverse_projection


def compute_projections(input_data, random_seeds, n_proj, size_proj, projection_mode):
    """ Computes `n_proj` projections of the whole input data over `size_proj` randomly chosen directions, using a
    given list of random seeds.

    :param input_data: high dimensional input data, shape=(batch_size, rows, cols, channels) #todo: type?
    :param random_seeds: random seeds for the projections, type=list
    :param n_proj: number of projections, type=int
    :param size_proj: size of a projection, type=int
    :return: random projections of input_data, type=np.ndarray, shape=(n_proj, batch_size, size, size, channels)
    """
    # this is needed to go from tf tensors to np arrays:
    sess = tf.Session()
    sess.as_default()

    print("Input shape: ", input_data.shape)
    print("\nComputing ",n_proj,"random projections in ",projection_mode,"mode: ")

    projections = []
    inverse_projections = []
    for proj_idx in range(n_proj):
        projection, inverse_projection = compute_single_projection(input_data, random_seeds[proj_idx], size_proj, projection_mode)
        projections.append(projection)
        inverse_projections.append(inverse_projection)
    projections = np.array(projections)
    inverse_projections = np.array(inverse_projections)
    print("Projected data dimensions:", projections.shape)

    return projections, inverse_projections


def compute_perturbations(input_data, inverse_projections):
    perturbations = np.copy(input_data)
    for channel in range(input_data.shape[3]):
        for inv_proj in inverse_projections:
            perturbations[:, :, :, channel] = np.add(perturbations[:, :, :, channel], inv_proj[:,:,:,channel])

    return perturbations


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


def plot_projections(image_data_list, cmap=None, test=False):#input_data,, ):#: random_seeds, n_proj, size_proj, projection_mode, test=False, ):
    """
    Plots input data in the first row, projected data in the second row and inverse projected data in the third row.
    By default it only takes the first 10 samples and the first projection.

    :param input_data: high dimensional input data
    :param random_seeds: list of random seeds for the projections
    :param n_proj: number of projections
    :param size_proj: size of a projection
    """

    n_images = 10
    fig, axs = plt.subplots(nrows=len(image_data_list), ncols=n_images, figsize=(10, 8))

    if image_data_list[0].shape[3] == 1:
        cmap = "gray"

    for group in range(len(image_data_list)):
        for im_idx in range(n_images):
            axs[group, im_idx].imshow(np.squeeze(image_data_list[group][im_idx]), cmap=cmap)

    if test is False:
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
