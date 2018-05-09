# -*- coding: utf-8 -*-
#

# Imports
import echotorch.nn
from .Transformer import Transformer


# Move to GPU
class ToCUDA(Transformer):
    """
    Move to GPU
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        # Super constructor
        super(ToCUDA, self).__init__()
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
        return 1
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
        return x.cuda()
    # end __call__

# end ToCUDA
