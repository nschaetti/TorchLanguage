# -*- coding: utf-8 -*-
#

# Imports
from .Transformer import Transformer


# Normalize a tensor
class Normalize(Transformer):
    """
    Normalize a tensor
    """

    # Constructor
    def __init__(self, mean, std, input_dim=1):
        """
        Constructor
        """
        # Super constructor
        super(Normalize, self).__init__()

        # Properties
        self.mean = mean
        self.std = std
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
        return (x - self.mean) / self.std
    # end __call__

# end Normalize
