# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Normalize a tensor on a dim
class NormalizeDim(Transformer):
    """
    Normalize a tensor on a dim
    """

    # Constructor
    def __init__(self, dim, mean=True, std=True, input_dim=1):
        """
        Constructor
        """
        # Super constructor
        super(NormalizeDim, self).__init__()

        # Properties
        self.dim = dim
        self.mean = mean
        self.std = std
        self.input_size = input_dim
    # end __init__

    ##############################################
    # Public
    ##############################################

    ##############################################
    # Private
    ##############################################

    # Sub mean
    def _sub_mean(self, x):
        """
        Sub mean
        :param x:
        :return:
        """
        # Means
        means = torch.mean(x, dim=self.dim)

        # For each entry
        for i in range(means.size(1)):
            x[0, i] -= float(means[0, i])
        # end for

        return x
    # end sub_mean

    # Div std
    def _div_std(self, x):
        """
        Div std
        :param x:
        :return:
        """
        # Means
        stds = torch.std(x, dim=self.dim)

        # For each entry
        for i in range(stds.size(1)):
            x[0, i] /= float(stds[0, i])
        # end for

        return x
    # end div_std

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
        if self.mean and self.std:
            return self._div_std(self._sub_mean(x))
        elif self.mean:
            return self._sub_mean(x)
        elif self.std:
            return self._div_std(x)
        else:
            return x
        # end if
    # end __call__

# end NormalizeDim
