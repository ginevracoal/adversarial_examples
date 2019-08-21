import numpy as np
import keras
from keras import backend as K
from keras.datasets import mnist
import pickle as pkl
import time
from sklearn.random_projection import GaussianRandomProjection
import os
import matplotlib.pyplot as plt

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
    # todo:docstring
    global x_train, y_train, x_test, y_test, input_shape, num_classes, data_format

    if dataset_name == "mnist":
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = preprocess_mnist(test=test)
    elif dataset_name == "cifar":
        x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_cifar(test=test)

    return x_train, y_train, x_test, y_test, input_shape, num_classes, data_format

######################
# random projections #
######################

def compute_single_projection_old(input_data, random_seed, size_proj):
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

    for im in range(input_data.shape[0]):
        projected_image = np.empty(shape=(size_proj * size_proj, input_data.shape[3]))
        for channel in range(input_data.shape[3]):
            projected_image[:, channel] = projector.fit_transform(flat_images[im, :, channel].reshape(1, -1))
        single_projection[im, :, :, :] = projected_image.reshape((size_proj, size_proj, input_data.shape[3]))

    return single_projection


def compute_single_projection(input_data, random_seed, size_proj):
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
        channel_projection = projector.fit_transform(flat_images[:, :, channel])\
                                      .reshape((input_data.shape[0], size_proj, size_proj))
        single_projection[:, :, :, channel] = channel_projection

    return single_projection

def compute_projections(input_data, random_seeds, n_proj, size_proj):
    """ Computes `n_proj` projections of the whole input data over `size_proj` randomly chosen directions, using a
    given list of random seeds.

    :param input_data: high dimensional input data
    :param random_seeds: list of random seeds for the projections
    :param n_proj: number of projections
    :param size_proj: size of a projection
    :return: np.array containing n_proj random projections on the data
    """

    print("Input shape: ", input_data.shape)
    print("\nComputing random projections: ")

    all_projections = np.empty(shape=(n_proj, input_data.shape[0], size_proj, size_proj, input_data.shape[3]))

    for proj_idx in range(n_proj):
        all_projections[proj_idx,:,:,:,:] = compute_single_projection(input_data=input_data, size_proj=size_proj,
                                                                      random_seed=random_seeds[proj_idx])
    print("Projected data dimensions:", all_projections.shape)

    # look at cifar projections
    #plot_projected_images(input_data=input_data, projected_data=all_projections, n_proj=n_proj, im_idxs=[1,2,6])

    return all_projections


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

    fig, axs = plt.subplots(len(im_idxs),n_proj+1,figsize=(10, 8))
    for im_idx, im in enumerate(im_idxs):
        axs[im_idx, 0].imshow(np.squeeze(input_data)[im_idx])
        for proj in range(n_proj):
            axs[im_idx, proj+1].imshow(np.squeeze(projected_data)[proj][im])

    # block plots at the end of code execution
    plt.show(block=False)
    input("Press ENTER to exit")
    exit()
