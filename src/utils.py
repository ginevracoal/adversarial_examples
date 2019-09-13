import numpy as np
import keras
from keras import backend as K
from keras.datasets import mnist
import pickle as pkl
import time
import os
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import StandardScaler

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

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

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
        # BUG: In the pickles I'm also saving the labels, so here I only take the data.
        # when loading only x_test it should become:
        # data = u.load()
        # todo: rigenerare i dati giusti su mnist o trasformare quelli che ho...
        data = u.load()#[0]
    if test is True:
        data = data[:TEST_SIZE]
    return data


##############
# plot utils #
##############


def plot_projections(image_data_list, cmap=None, test=False):
    """
    Plots the first `n_images` images of each element in image_data_list, on different rows.

    :param image_data_list: list of sets of images to plot
    :param cmap: colormap  = gray or None
    :param test: if True it does not hang on the image
    """

    n_images = 5
    fig, axs = plt.subplots(nrows=len(image_data_list), ncols=n_images, figsize=(n_images, 1.5*len(image_data_list)))
    # fig.suptitle("CIFAR10 projection", fontsize=20, y=0.95)
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


############

def compute_angle(v1, v2):
    """ Compute the angle between two numpy arrays, eventually flattening them if multidimensional. """
    if len(v1) != len(v2): raise ValueError("\nYou cannot compute the angle between vectors with different dimensions.")
    v1 = v1.flatten()
    v2 = v2.flatten()
    return math.acos(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)) )

def compute_variances(x,y):
    """
    Compute within-class and between-class variances on the given data.
    :param x: input data, type=np.ndarray, shape=(n_samples, n_features)
    :param y: data labels, type=np.ndarray, shape=(n_samples, n_classes)
    :return: average within-class variance, average between-class variance
    """

    # reshape and standardize data
    n_features =  x.shape[1]*x.shape[2]*x.shape[3]
    x = x.reshape(len(x),n_features)
    x = (x - np.mean(x))/ np.std(x)

    # compute mean and class mean
    mu = np.mean(x, axis=0).reshape(n_features,1)
    y_true = np.argmax(y, axis=1)
    mu_classes = []
    for i in range(10):
        mu_classes.append(np.mean(x[np.where(y_true == i)], axis=0))
    mu_classes = np.array(mu_classes).T

    # compute scatter matrices
    data_SW = []
    Nc = []
    for i in range(10):
        a = np.array(x[np.where(y_true == i)] - mu_classes[:, i].reshape(1, n_features))
        data_SW.append(np.dot(a.T, a))
        Nc.append(np.sum(y_true == i))
    SW = np.sum(data_SW, axis=0)
    SB = np.dot(Nc * np.array(mu_classes - mu), np.array(mu_classes - mu).T)
    SW_min = np.min(SW)
    SW_mean = np.mean(SW)
    SW_max = np.max(SW)
    SB_min = np.min(SB)
    SB_mean = np.mean(SB)
    SB_max = np.max(SB)
    print("\nWithin-class variance: min=",SW_min, ", avg=", SW_mean,", max=", SW_max)
    print("Between-class variance: min=",SB_min, ", avg=", SB_mean,", max=", SB_max)

    return SW_mean, SB_mean