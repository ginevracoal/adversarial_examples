from utils import *
from random_ensemble import RandomEnsemble
import sys
from adversarial_classifier import *


def main(dataset_name, test, proj_idx, size_proj):

    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = load_dataset(dataset_name, test)

    model = RandomEnsemble(input_shape=input_shape, num_classes=num_classes, n_proj=None, size_proj=size_proj,
                           data_format=data_format, dataset_name=dataset_name)
    model.train_single_projection(x_train=x_train, y_train=y_train, batch_size=model.batch_size,
                                  epochs=model.epochs, idx=proj_idx, save=True)


if __name__ == "__main__":

    try:
        dataset_name = sys.argv[1]
        test = eval(sys.argv[2])
        proj_idx = int(sys.argv[3])
        size_proj_list = list(map(int, sys.argv[4].strip('[]').split(',')))

    except IndexError:
        dataset_name = input("\nChoose a dataset ("+DATASETS+"): ")
        test = input("\nDo you just want to test the code? (True/False): ")
        proj_idx = input("\nChoose the projection idx. ")
        size_proj_list = input("\nChoose size for the projection. ")

    for size_proj in size_proj_list:
        K.clear_session()
        main(dataset_name=dataset_name, test=test, proj_idx=proj_idx, size_proj=size_proj)


