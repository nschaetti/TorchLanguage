# -*- coding: utf-8 -*-
#

# Imports
import echotorch.nn
from .Transformer import Transformer


# Reshape input tensor
class Reshape(Transformer):
    """
    Reshape input tensor
    """

    # Constructor
    def __init__(self, view):
        """
        Constructor
        :param model: Feature selection model.
        """
        # Super constructor
        super(Reshape, self).__init__()

        # Properties
        self.view = view
    # end __init__

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
        return self.view[-1]
    # end input_dim

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, x):
        """
        Convert a string to a ESN input
        :param x: Tensor to transform
        :return: Tensor of word vectors
        """
        return x.view(self.view)
    # end __call__

# end Reshape
