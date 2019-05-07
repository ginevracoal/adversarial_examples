import keras
from keras.datasets import mnist
from keras import backend as K


NUM_CLASSES = 10
IMG_COLS = 28
IMG_ROWS = 28
MIN = 0
MAX = 255


def preprocess_mnist(img_rows=IMG_ROWS, img_cols=IMG_COLS):
    """Preprocess mnist dataset for keras training

    :param img_rows, img_cols: input image dimensions
    """

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
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test, input_shape, NUM_CLASSES


def compute_random_projections(input_data, n_proj, dim_proj=None):
    """ Computes m projections of the whole input data over k randomly chosen directions.

    :param input_data: full dimension input data
    :param n_proj: number of projections
    :param dim_proj: dimensionality of a projection
    :param random_state: pseudo random number generator
    :return: array containing the m projections
    """

    # TODO: non funziona il metodo di johns-lind
    if dim_proj is None:
        dim_proj = 'auto'

    print(input_data.shape)

    projected_data = []
    for i in range(n_proj):
        projected_data.append(GaussianRandomProjection(n_components=dim_proj).fit_transform(input_data))
    projected_data = np.array(projected_data)

    print(projected_data.shape)
    return projected_data

