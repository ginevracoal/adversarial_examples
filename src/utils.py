import numpy as np
import keras
from keras import backend as K
from keras.datasets import mnist
import pickle as pkl
import time
from sklearn.random_projection import GaussianRandomProjection


IMG_ROWS = 28
IMG_COLS = 28
MIN = 0
MAX = 255
TEST_SIZE = 100
RESULTS = "../results/"


def preprocess_mnist(test=False, img_rows=IMG_ROWS, img_cols=IMG_COLS):
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

    if test is True:
        return x_train[:TEST_SIZE], y_train[:TEST_SIZE], x_test[:TEST_SIZE], y_test[:TEST_SIZE], input_shape, 10
    else:
        return x_train, y_train, x_test, y_test, input_shape, 10


def compute_projections(input_data, random_seeds, n_proj, size_proj=None):
    """ Computes `n_proj` projections of the whole input data over `size_proj` randomly chosen directions, using a
    given projector function `projector`.

    :param input_data: full dimension input data
    :type input_data: numpy array
    :param projector: projector function
    :type projector: GaussianRandomProjection object
    :param n_proj: number of projections
    :type n_proj: int
    :param size_proj: size of a projection
    :type size_proj: int
    :param random_state: pseudo random number generator
    :type random_state: int
    :return: array containing m random projections
    """

    print("\nComputing random projections.")

    # TODO: non funziona il metodo di johns-lind
    if size_proj is None:
        size_proj = 'auto'

    print("Input shape: ", input_data.shape)
    flat_images = input_data.reshape(input_data.shape[0], input_data.shape[1]*input_data.shape[2]*input_data.shape[3])

    projected_data = []
    for i in range(n_proj):
        # cannot use list comprehension on GaussianRandomProjection objects
        projector = GaussianRandomProjection(n_components=size_proj * size_proj, random_state=random_seeds[i])
        projected_data.append(projector.fit_transform(flat_images))

    projected_data = np.array(projected_data).reshape(n_proj, input_data.shape[0], size_proj, size_proj, 1)

    print("Projected data shape:", projected_data.shape)
    return projected_data


# Pickle utils


def save_to_pickle(data, filename):
    """ saves data to pickle """
    with open(RESULTS+time.strftime('%Y-%m-%d')+"/"+filename, 'wb') as f:
        pkl.dump(data, f)


def load_from_pickle(path):
    """ loads data from pickle """
    with open(path, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    return data

##############
# DEPRECATED #
##############


class List(list):
    # TODO: remove this
    """
    A subclass of list that can accept additional attributes.
    Should be able to be used just like a regular list.
    """
    def __new__(self, *args, **kwargs):
        return super(List, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self
