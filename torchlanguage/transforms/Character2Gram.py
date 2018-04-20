# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer
import numpy as np


# Transform text to character 2-gram
class Character2Gram(Transformer):
    """
    Transform text to character 2-grams
    """

    # Constructor
    def __init__(self, overlapse=True):
        """
        Constructor
        """
        # Super constructor
        super(Character2Gram, self).__init__()

        # Properties
        self.overlapse = overlapse
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
        # List of character
        return self._transform(text)
    # end convert

    ##############################################
    # Private
    ##############################################

    # Transform
    def _transform(self, x):
        """
        Transform input
        :param x:
        :return:
        """
        # Step
        if self.overlapse:
            step = 1
            last = 1
        else:
            step = 2
            last = 0
        #  end if

        # List of character to 2grams
        return [x[i:i + 2] for i in np.arange(0, len(x) - last, step)]
    # end _transform

# end Character2Gram
