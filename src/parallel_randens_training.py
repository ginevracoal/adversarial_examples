from utils import *
from random_ensemble import RandomEnsemble
import sys

def compute_single_projection(input_data, random_seed, size_proj):
    """ Computes one projection of the whole input data over `size_proj` randomly chosen directions with Gaussian
        matrix entries sampling.

    :param input_data: full dimension input data
    :param random_seed: list of seeds for the projections
    :param size_proj: size of a projection
    :return: array containing all random projections of the data, based on the given seed
    """

    print("\nProjecting the whole data over a single subspace.")

    print("Input shape: ", input_data.shape)
    flat_images = input_data.reshape(input_data.shape[0], input_data.shape[1]*input_data.shape[2]*input_data.shape[3])

    projector = GaussianRandomProjection(n_components=size_proj * size_proj, random_state=random_seed)
    projected_data = projector.fit_transform(flat_images)

    # reshape in matrix form
    projected_data = np.array(projected_data).reshape(input_data.shape[0], size_proj, size_proj, 1)

    print("Projected data shape:", projected_data.shape)
    return projected_data


def main(dataset_name, test, proj_idx, size_proj):

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)

    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes,
                           n_proj=None, size_proj=size_proj, data_format=data_format, dataset_name=dataset_name)

    x_train_projected = compute_single_projection(input_data=x_train, random_seed=model.random_seeds[proj_idx],
                                                  size_proj=model.size_proj)

    model.train_single_projection(x_train_projected=x_train_projected, y_train=y_train, batch_size=model.batch_size,
                                  epochs=model.epochs, idx=proj_idx, save=False)


if __name__ == "__main__":

    try:
        dataset_name = sys.argv[1]
        test = sys.argv[2]
        proj_idx = int(sys.argv[3])
        size_proj = int(sys.argv[4])

    except IndexError:
        dataset_name = input("\nChoose a dataset.")
        test = input("\nDo you just want to test the code?")
        proj_idx = input("\nChoose the projection idx.")
        size_proj = input("\nChoose size for the projection.")

    main(dataset_name=dataset_name, test=test, proj_idx=proj_idx, size_proj=size_proj)


