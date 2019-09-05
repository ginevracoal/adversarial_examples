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


def tf_flat_projection(input_data, random_seed, size_proj):
    """ Computes a projection of the whole input data flattened over channels and also computes the inverse projection.
    It samples `size_proj` random directions for the projection using the given `random_seed`.

    :param input_data: high dimensional input data, type=tf.tensor, shape=(batch_size, rows, cols, channels)
    :param random_seed: projection seed, type=int
    :param size_proj: size of a projection, type=int
    :return:
    :param projection: random projection of input_data, type=tf.tensor,
                       shape=(batch_size, size_proj, size_proj, channels)
    :param projection: inverse projection of input_data given by the Moore-Penrose pseudoinverse of the projection
                       matrix, type=tf.tensor, shape=(batch_size, size, size, channels)

    """
    input_data = tf.cast(input_data, tf.float32)
    batch_size, rows, cols, channels = input_data.get_shape().as_list()
    n_features = rows * cols * channels
    n_components = size_proj * size_proj * channels

    # projection matrices
    projector = GaussianRandomProjection(n_components=n_components, random_state=random_seed)
    proj_matrix = np.float32(projector._make_random_matrix(n_components, n_features))
    pinv = np.linalg.pinv(proj_matrix)

    # compute projections
    flat_images = tf.reshape(input_data, shape=[batch_size, n_features])
    projection = tf.matmul(a=flat_images, b=proj_matrix, transpose_b=True)
    inverse_projection = tf.matmul(a=projection, b=pinv, transpose_b=True)

    # reshape
    projection = tf.reshape(projection, shape=tf.TensorShape([batch_size, size_proj, size_proj, channels]))
    inverse_projection = tf.reshape(inverse_projection, shape=tf.TensorShape([batch_size, rows, cols, channels]))

    # inverse_projection = tf.cast(inverse_projection, tf.float32)
    inverse_projection = mod_invproj(tf.cast(inverse_projection, tf.float32))
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
    sess = tf.Session()
    sess.as_default()

    projection, inverse_projection = tf_flat_projection(input_data, random_seed, size_proj)
    projection = projection.eval(session=sess)
    inverse_projection = inverse_projection.eval(session=sess)
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
    n_proj = inverse_projections.shape[0]

    perturbations = np.empty(input_data.shape, dtype=float)
    for inverse_projection in inverse_projections:
        perturbations = np.add(perturbations / n_proj, inverse_projection)

    augmented_inputs = np.add(input_data, perturbations)
    augmented_inputs = [mod_augmented_inputs(x.astype(float)) for x in augmented_inputs]
    augmented_inputs = np.array(augmented_inputs)

    return perturbations, augmented_inputs

# TESTING SOME VARIANTS

def mod_invproj(x):
    return x

def mod_pertubations(x):
    return x

def mod_augmented_inputs(x):
    return x

def rescale(x):
    return x*255.0

def normalize(x):
    """
    normalize rgb images.
    :param x: input image, shape=(rows, cols, channels), dtype=float
    :return: normalized image
    """
    return (x-128) / 128

def to_rgb(x):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    min = tf.math.reduce_min(x).eval(session=sess)
    max = tf.math.reduce_max(x).eval(session=sess)
    return 255.0 * tf.div(tf.subtract(x, min), tf.subtract(max, min))
