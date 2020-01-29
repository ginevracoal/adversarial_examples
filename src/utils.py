import numpy as np
import keras
from keras import backend as K
from keras.datasets import mnist, fashion_mnist
import pickle as pkl
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
import tensorflow as tf
import torch
from directories import *
from pandas import DataFrame

TEST_SIZE = 20


def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExecution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))


######################
# data preprocessing #
######################

def load_fashion_mnist(img_rows=28, img_cols=28, n_samples=None):
    print("\nLoading fashion mnist.")

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if n_samples:
        x_train = x_train[:n_samples]
        y_train = y_train[:n_samples]
        x_test = x_test[:n_samples]
        y_test = y_test[:n_samples]

    num_classes = 10
    data_format = 'channels_last'

    print('x_train shape:', x_train.shape, '\nx_test shape:', x_test.shape)
    return x_train, y_train, x_test, y_test, input_shape, num_classes, data_format

def preprocess_mnist(test, img_rows=28, img_cols=28, n_samples=None):
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

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if n_samples:
        x_train = x_train[:n_samples]
        y_train = y_train[:n_samples]
        x_test = x_test[:n_samples]
        y_test = y_test[:n_samples]
    else:
        if test:
            x_train = x_train[:TEST_SIZE]
            y_train = y_train[:TEST_SIZE]
            x_test = x_test[:TEST_SIZE]
            y_test = y_test[:TEST_SIZE]

    num_classes = 10
    data_format = 'channels_last'

    # # swap channels
    # x_train = np.zeros((x_train.shape[0], img_rows, img_cols, 1))
    # x_train = np.rollaxis(x_train, 3, 1)
    # x_test = np.zeros((x_test.shape[0], img_rows, img_cols, 1))
    # x_test = np.rollaxis(x_test, 3, 1)
    # data_format = "channels_first"
    # input_shape = (1, img_rows, img_cols)

    print('x_train shape:', x_train.shape, '\nx_test shape:', x_test.shape)
    return x_train, y_train, x_test, y_test, input_shape, num_classes, data_format


def _onehot(integer_labels):
    """Return matrix whose rows are onehot encodings of integers."""
    n_rows = len(integer_labels)
    n_cols = integer_labels.max() + 1
    onehot = np.zeros((n_rows, n_cols), dtype='uint8')
    onehot[np.arange(n_rows), integer_labels] = 1
    return onehot

def onehot_to_labels(y):
    if type(y) is np.ndarray:
        return np.argmax(y, axis=1)
    elif type(y) is torch.Tensor:
        return torch.max(y, 1)[1]

def load_cifar(test, data, n_samples=None):
    """Return train_data, train_labels, test_data, test_labels
    The shape of data is 32 x 32 x3"""
    x_train = None
    y_train = []

    data_dir=str(data)+'cifar-10/'

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

    if n_samples:
        x_train = x_train[:n_samples]
        y_train = y_train[:n_samples]
        x_test = x_test[:n_samples]
        y_test = y_test[:n_samples]
    else:
        if test:
            x_train = x_train[:TEST_SIZE]
            y_train = y_train[:TEST_SIZE]
            x_test = x_test[:TEST_SIZE]
            y_test = y_test[:TEST_SIZE]

    return x_train, _onehot(y_train), x_test, _onehot(y_test), input_shape, num_classes, data_format


def load_dataset(dataset_name, test, data=DATA_PATH, n_samples=None):
    """
    Load dataset.
    :param dataset_name: choose between "mnist" and "cifar"
    :param test: If True only loads the first 100 samples
    """
    # global x_train, y_train, x_test, y_test, input_shape, num_classes, data_format

    if dataset_name == "mnist":
        return preprocess_mnist(test=test, n_samples=n_samples)
    elif dataset_name == "cifar":
        return load_cifar(test=test, data=data, n_samples=n_samples)
    elif dataset_name == "fashion_mnist":
        return load_fashion_mnist(n_samples=n_samples)
    else:
        raise ValueError("\nWrong dataset name.")


################
# Pickle utils #
################


def save_to_pickle(data, relative_path, filename):
    """ saves data to pickle """

    filepath = relative_path + filename
    print("\nSaving pickle: ", filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pkl.dump(data, f)


def unpickle(file):
    """ Load byte data from file"""
    with open(file, 'rb') as f:
        data = pkl.load(f, encoding='latin-1')
    return data


def load_from_pickle(path, test=False):
    """ loads data from pickle containing: x_test, y_test."""
    print("\nLoading from pickle: ",path)
    with open(path, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    if test is True:
        data = data[:TEST_SIZE]
    return data


##############
# plot utils #
##############

def covariance_eigendec(np_matrix):
    print("\nmatrix[rows=vars, cols=obs] = \n", np_matrix)
    C = np.cov(np_matrix)
    print("\ncovariance matrix = \n", C)

    eVe, eVa = np.linalg.eig(C)

    plt.scatter(np_matrix[:, 0], np_matrix[:, 1])
    for e, v in zip(eVe, eVa.T):
        plt.plot([0, 3 * np.sqrt(e) * v[0]], [0, 3 * np.sqrt(e) * v[1]], 'k-', lw=2)
    plt.title('Transformed Data')
    plt.axis('equal')
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    plt.savefig(RESULTS+"covariance.png")


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


def compute_covariance_matrices(x,y):
    """
    Compute within-class and between-class variances on the given data.
    :param x: input data, type=np.ndarray, shape=(n_samples, n_features)
    :param y: data labels, type=np.ndarray, shape=(n_samples, n_classes)
    :return: average within-class variance, average between-class variance
    """
    standardize = lambda x: (x - np.mean(x))/ np.std(x)
    normalize = lambda x: (x - np.min(x))/ (np.max(x)-np.min(x))

    # reshape and standardize data
    n_features =  x.shape[1]
    x = x.reshape(len(x),n_features)
    x = standardize(x)

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

    SW = normalize(SW)
    SB = normalize(SB)
    print("\nWithin-class avg normalized variance:", np.mean(SW))
    print("Between-class avg normalized variance:", np.mean(SB))

    return SW, SB


def compute_distances(x1,x2,ord):
    """
    Computes min, avg and max distances between the inputs and their perturbations
    :param x1: input points, shape=(n_samples, rows, cols, channels), type=np.ndarray
    :param x2: perturbations, shape=(n_samples, rows, cols, channels), type=np.ndarray
    :param ord: norm order for np.linalg.norm
    :return: min, average, max distances between all couples of points, type=dict
    """
    if x1.shape != x2.shape:
        raise ValueError("\nThe arrays need to have the same shape.")
    flat_x1 = x1.reshape(x1.shape[0], np.prod(x1.shape[1:]))
    flat_x2 = x2.reshape(x2.shape[0], np.prod(x2.shape[1:]))
    min = np.min([np.linalg.norm(flat_x1[idx] - flat_x2[idx], ord=ord) for idx in range(len(x1))])
    mean = np.mean([np.linalg.norm(flat_x1[idx] - flat_x2[idx], ord=ord) for idx in range(len(x1))])
    max = np.max([np.linalg.norm(flat_x1[idx] - flat_x2[idx], ord=ord) for idx in range(len(x1))])
    return {"min":min,"mean": mean,"max": max}


def _set_session(device, n_jobs):
    """
     Initialize tf session on device.
    :param device:
    :param n_jobs:
    :return:
    """


    # from keras.backend.tensorflow_backend import set_session
    sess = tf.Session()
    # print(device_lib.list_local_devices())
    if device == "gpu":
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        # config.allow_soft_placement = True
        # config.log_device_placement = True  # to log device placement (on which device the operation ran)
        config.gpu_options.per_process_gpu_memory_fraction = 1/n_jobs
        sess = tf.compat.v1.Session(config=config)

    keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
    sess.run(tf.global_variables_initializer())
    # print("check cuda: ", tf.test.is_built_with_cuda())
    # print("check gpu: ", tf.test.is_gpu_available())
    return sess

# Plot utils

def plot_heatmap(columns, path, filename, xlab=None, ylab=None, title=None, yticks=None):
    columns = np.array(columns)
    # print(columns.shape)
    fig, ax = plt.subplots(figsize=(15, 6), dpi=400)
    sns.heatmap(columns, ax=ax)
    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    if yticks:
        ax.set_yticklabels(yticks)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path+filename)


def plot_loss_accuracy(dict, path):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8))
    ax1.plot(dict['loss'])
    ax1.set_title("loss")
    ax2.plot(dict['accuracy'])
    ax2.set_title("accuracy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)


def violin_plot(data, path, filename, xlab=None, ylab=None, title=None, yticks=None):
    fig, axes = plt.subplots(figsize=(15, 6))
    # sns.set(style="whitegrid")
    sns.violinplot(data=data, ax=axes, orient='v')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + filename)



