# -*- coding: utf-8 -*-
#

# Imports


# Do a k-fold cross validation on a data set
class CrossValidation(object):
    """
    Do K-fold cross validation on a data set
    """

    # Constructor
    def __init__(self, dataset, k=10, train=True):
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
    # end __init__

    def __getitem__(self, item):
        pass
    # end __getitem__

# end CrossValidation

