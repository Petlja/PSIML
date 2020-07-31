import numpy as np
import pandas as pd

from Constants import c_seed, c_trainPortion, c_validationPortion, c_testPortion


def SplitDataset(dataset, seed=c_seed, trainPortion=c_trainPortion, validationPortion=c_validationPortion, testPortion=c_testPortion):
    """
    Splits the dataset into training, validation and test sets

    Args:
        dataset (pandas.DataFrame): dataset that should be split
        seed (int): seed of the random generator (necessary to achieve reproducible results)
        trainPortion (double): portion of the training set
        validationPortion (double): portion of the validation set
        testPortion (double): portion of the test set
    Returns:
        dataset (pandas.DataFrame): dataset with additional column denoting the set the row belongs to, after the split
    """
    # list of items in the dataset
    item_list = []

    # transform each item to dictionary to add it to list of items
    for _, row in dataset.iterrows():
        item_list.append(row.to_dict())

    # set the random seed
    np.random.seed(seed)

    # shuffle the dataset (
    # note: np.random.shuffle() function takes ndarray as input parameter
    np.random.shuffle(item_list)

    # total number of items in the dataset
    n_total = len(item_list)
    # number of items in the training set
    n_train = int(trainPortion * n_total)
    # number of items in the validation set
    n_validation = int(validationPortion * n_total)
    # number of items in test set
    n_test = int(testPortion * n_total)

    # Give data point a split attribute

    # for items that should belong to training set
    for item in item_list[0:n_train]:
        item["split"] = "train"

    # for items that should belong to validation set
    for item in item_list[n_train:n_train+n_validation]:
        item["split"] = "validation"

    # for items that should belong to test set
    for item in item_list[n_train+n_validation:n_train+n_validation+n_test]:
        item["split"] = "test"

    dataset = pd.DataFrame(item_list)

    return dataset