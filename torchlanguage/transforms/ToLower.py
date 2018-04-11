# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Text to lower case
class ToLower(Transformer):
    """
    Text to lower case
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        # Super constructor
        super(ToLower, self).__init__()
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
        return 1
    # end input_dim

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, text):
        """
        Convert a string to a ESN input
        :param text: Text to convert
        :return: Tensor of word vectors
        """
        return text.lower()
    # end convert

# end ToLower
