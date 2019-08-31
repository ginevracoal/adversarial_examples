import numpy as np
import tensorflow as tf
from sklearn.random_projection import GaussianRandomProjection
from utils import rgb2gray

MIN = 0
MAX = 255
TEST_SIZE = 100
RESULTS = "../results/"


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
    # sess = tf.Session()
    # sess.as_default()
    # input_data = tf.cast(input_data, tf.float32)

    print("Input shape: ", input_data.shape)
    print("\nComputing ",n_proj,"random projections in ",projection_mode,"mode: ")

    projections = []
    inverse_projections = []
    for proj_idx in range(n_proj):
        projection, inverse_projection = compute_single_projection(input_data, random_seeds[proj_idx],
                                                                        size_proj, projection_mode)
        projections.append(projection)
        inverse_projections.append(inverse_projection)
    projections = np.array(projections)
    inverse_projections = np.array(inverse_projections)

    print("Projected data dimensions:", projections.shape)
    # projections = tf.convert_to_tensor(projections)
    # inverse_projections = tf.convert_to_tensor(inverse_projections)
    # print(projections, inverse_projections)
    return projections, inverse_projections


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


def flat_projection(input_data, random_seed, size_proj):
    """ Computes a projection of the whole input data flattened over channels and also computes the inverse projection.
    It samples `size_proj` random directions for the projection using the given random_seed.

    :param input_data: high dimensional input data, type=np.ndarray, shape=(n_samples,rows,cols,channels)
    :param random_seed: projection seed, type=int
    :param size_proj: size of a projection, type=int
    :return:
    :param projection: random projection of input_data, type=np.ndarray, shape=(n_samples,size_proj,size_proj,channels)
    :param inverse_projection: inverse projection of input_data given by the pseudoinverse of the projection matrix,
                               type=np.ndarray, shape=(n_samples,rows,cols,channels)
    """
    samples, rows, cols, channels = input_data.shape
    # this is needed to go from tf tensors to np arrays:
    sess = tf.Session()
    sess.as_default()

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

    :param input_data: high dimensional input data, type=np.ndarray, shape=(n_samples,rows,cols,channels)
    :param random_seed: projection seed, type=int
    :param size_proj: size of a projection, type=int
    :return:
    :param projection: random projection of input_data, type=np.ndarray, shape=(n_samples,size_proj,size_proj,channels)
    :param inverse_projection: inverse projection of input_data given by the pseudoinverse of the projection matrix,
                               type=np.ndarray, shape=(n_samples,rows,cols,channels)
    """

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
    """ Transforms input_data into rgb representation and calls flat_projection on it.
    :param input_data: high dimensional input data, type=np.ndarray, shape=(n_samples,rows,cols,channels)
    :param random_seed: projection seed, type=int
    :param size_proj: size of a projection, type=int
    :return:
    :param projection: random projection of input_data, type=np.ndarray, shape=(n_samples,size_proj,size_proj,channels)
    :param inverse_projection: inverse projection of input_data given by the pseudoinverse of the projection matrix,
                               type=np.ndarray, shape=(n_samples,rows,cols,channels)
    """
    samples, rows, cols, channels = input_data.shape
    greyscale_data = np.array([rgb2gray(rgb_im) for rgb_im in input_data]).reshape((samples, rows, cols, 1))
    return flat_projection(greyscale_data, random_seed, size_proj)


def compute_perturbations(input_data, inverse_projections):
    """
    Compute input_data perturbations by adding up inverse_projections on separated color channels.
    :param input_data: input data, type=np.ndarray, shape=(n_samples, rows, cols, channels)
    :param inverse_projections: inverse projections of the input data, type=np.ndarray,
                                shape=(n_proj, n_samples, rows, cols, channels)
    :return: perturbations, type=np.ndarray, shape=(n_samples, rows, cols, channels)
             augmented_inputs are computed by adding perturbations to the inputs, type=np.ndarray,
             shape=(n_samples, rows, cols, channels)
    """

    perturbations = np.empty(input_data.shape)
    for inverse_projection in inverse_projections:
        for channel in range(input_data.shape[3]):
            perturbations[:, :, :, channel] = np.add(perturbations[:, :, :, channel], inverse_projection[:,:,:,channel])

    augmented_inputs = np.add(input_data,perturbations)

    return perturbations, augmented_inputs
