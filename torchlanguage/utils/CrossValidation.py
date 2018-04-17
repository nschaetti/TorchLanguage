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
    def __init__(self, dataset, k=10, train=True, fold=0):
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
        self.fold = fold
        self.indexes = np.arange(0, len(dataset))
        self.fold_length = int(math.floor(len(dataset) / k))
    # end __init__

    ###################################
    # OVERRIDE
    ###################################

    # Dataset size
    def __len__(self):
        """
        Dataset size
        :return:
        """
        if self.train:
            return self.fold_length * (self.k - 1)
        else:
            return self.fold_length
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
        start_index = self.fold * self.k
        test_set = self.indexes[start_index:start_index+self.fold_length]
        train_set = np.remove(self.indexes, test_set)

        # Train/test
        if self.train:
            return self.dataset[train_set[item]]
        else:
            return self.dataset[test_set[item]]
        # end if
    # end __getitem__

# end CrossValidation

