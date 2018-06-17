# -*- coding: utf-8 -*-
#

# Imports
import math
from torch.utils.data.dataset import Dataset
import numpy as np


# Do a k-fold cross validation on a data set
class CrossValidation(Dataset):
    """
    Do K-fold cross validation on a data set
    """

    # Constructor
    def __init__(self, dataset, k=10, train=True, fold=0, train_size=1.0):
        """
        Constructor
        :param dataset: The target data set
        :param k: Nnumber of fold
        :param train: Return training or test set?
        """
        # Properties
        self.dataset = dataset
        self.k = k
        self.train = train
        self.train_size = train_size
        self.fold = fold
        self.folds, self.fold_sizes, self.indexes = self._create_folds(self.k)
    # end __init__

    ###################################
    # PUBLIC
    ###################################

    # Next fold
    def next_fold(self):
        """
        Next fold
        :return:
        """
        self.fold += 1
    # end next_fold

    # Set fold
    def set_fold(self, fold):
        """
        Set fold
        :param fold:
        :return:
        """
        self.fold = fold
    # end set_fold

    # Set size
    def set_size(self, size):
        """
        Set size
        :param size:
        :return:
        """
        self.train_size = size
    # end set_size

    ###################################
    # PRIVATE
    ###################################

    # Create folds
    def _create_folds(self, k):
        """
        Create folds
        :param indexes:
        :return:
        """
        # Indexes
        indexes = np.arange(0, len(self.dataset))

        # Dataset length
        length = len(indexes)

        # Division and rest
        division = int(math.floor(length / k))
        reste = length - division * k
        reste_size = k - reste

        # Folds size
        fold_sizes = [division+1] * (reste) + [division] * (reste_size)

        # Folds
        folds = list()
        start = 0
        for i in range(k):
            folds.append(indexes[start:start+fold_sizes[i]])
            start += fold_sizes[i]
        # end for

        return folds, fold_sizes, indexes
    # end _create_folds

    ###################################
    # OVERRIDE
    ###################################

    # Dataset size
    def __len__(self):
        """
        Dataset size
        :return:
        """
        # Test length
        test_length = self.fold_sizes[self.fold]
        train_length = len(self.dataset) - test_length

        if self.train:
            return int(train_length * self.train_size)
        else:
            return test_length
        # end if
    # end __len__

    # Get item
    def __getitem__(self, item):
        """
        Get item
        :param item:
        :return:
        """
        # Get target set
        test_set = self.folds[self.fold]
        indexes_copy = self.indexes.copy()
        train_set = np.delete(indexes_copy, test_set)
        train_length = len(self.dataset) - len(test_set)
        train_length = int(train_length * self.train_size)
        train_set = train_set[:train_length]

        # Train/test
        if self.train:
            return self.dataset[train_set[item]]
        else:
            return self.dataset[test_set[item]]
        # end if
    # end __getitem__

# end CrossValidation

