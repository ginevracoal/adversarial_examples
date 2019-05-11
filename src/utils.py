import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import keras
from keras import backend as K
from keras.datasets import mnist

NUM_CLASSES = 10
IMG_COLS = 28
IMG_ROWS = 28
MIN = 0
MAX = 255


def preprocess_mnist(img_rows=IMG_ROWS, img_cols=IMG_COLS):
    """Preprocess mnist dataset for keras training

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
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test, input_shape, NUM_CLASSES


def compute_projections(input_data, projector, n_proj, size_proj=None):

    """ Computes m projections of the whole input data over k randomly chosen directions.
    # TODO: change docstring
    :param input_data: full dimension input data
    :param n_proj: number of projections
    :param size_proj: size of a projection
    :param random_state: pseudo random number generator
    :return: array containing m random projections
    """
    print("\nComputing random projections.")

    # TODO: non funziona il metodo di johns-lind
    if size_proj is None:
        size_proj = 'auto'

    # TODO: dim proj deve essere inferiore a 28 e superiore a (vedi struttura layers)

    flat_images = input_data.reshape(input_data.shape[0], input_data.shape[1]*input_data.shape[2]*input_data.shape[3])
    print("Input shape: ", input_data.shape)

    projected_data = [projector.fit_transform(flat_images) for i in range(n_proj)]
    projected_data = np.array(projected_data).reshape(n_proj, input_data.shape[0], size_proj, size_proj, 1)

    print("Output shape:", projected_data.shape)
    print("->", len(projected_data), "random projections of", projected_data.shape[1], "images whose shape is",
          projected_data.shape[2:])

    return projected_data


class List(list):
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
