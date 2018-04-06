# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer
import numpy as np


# Transform text to character 3-gram
class Character3Gram(Transformer):
    """
    Transform text to character 3-grams
    """

    # Constructor
    def __init__(self, uppercase=False, overlapse=True):
        """
        Constructor
        """
        # Properties
        self.uppercase = uppercase
        self.overlapse = overlapse

        # Super constructor
        super(Character3Gram, self).__init__()
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
    # Private
    ##############################################

    # To upper
    def to_upper(self, gram):
        """
        To upper
        :param gram:
        :return:
        """
        if not self.uppercase:
            return gram.lower()
        # end if
        return gram
    # end to_upper

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
        # Step
        if self.overlapse:
            step = 1
            last = 2
        else:
            step = 3
            last = 0
        #  end if

        # List of character to 3 grams
        return [self.to_upper(text[i:i+3]) for i in np.arange(0, len(text)-last, step)]
    # end convert

# end Character3Gram
