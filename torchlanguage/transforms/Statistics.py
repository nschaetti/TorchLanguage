# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Gather statistics about tensors
class Statistics(Transformer):
    """
    Gather statistics about tensors
    """

    # Constructor
    def __init__(self, input_dim=1):
        """
        Constructor
        """
        # Super constructor
        super(Statistics, self).__init__()

        # Properties
        self.sum = 0
        self.total = 0
        self.count = 0
        self.sd = 0
        self.max = -2000000000
        self.min = 2000000000
        self.input_size = input_dim
    # end __init__

    ##############################################
    # Public
    ##############################################

    ##############################################
    # Properties
    ##############################################

    # Get the number of inputs
    @property
    def input_dim(self):
        """
        Get the number of inputs.
        :return: The input size.
        """
        return self.input_size
    # end input_dim

    # Get the average
    @property
    def mean(self):
        """
        Get the average
        :return:
        """
        return self.sum / self.total
    # end mean

    # Standard deviation
    @property
    def std(self):
        """
        Standard deviation
        :return:
        """
        return self.sd / self.count
    # end std

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, x):
        """
        Convert a string to a ESN input
        :param x: Tensor to normalize
        :return: Tensor of word vectors
        """
        self.total += x.nelement()
        self.count += 1.0
        self.sum += torch.sum(x)
        self.sd += torch.std(x)
        if torch.max(x) > self.max:
            self.max = torch.max(x)
        # end if
        if torch.min(x) < self.min:
            self.min = torch.min(x)
        # end if
        return x
    # end __call__

# end Statistics
