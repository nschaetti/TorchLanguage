# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Create random samples of a given size from a tensor of one or two dimensions
class RandomSamples(Transformer):
    """
    Create random samples of a given size from a tensor of one or two dimensions
    """

    # Constructor
    def __init__(self, n_samples, sample_size):
        """
        Constructor
        :param n_samples: Number of samples
        :param sample_size: Samples size
        """
        # Super constructor
        super(RandomSamples, self).__init__()

        # Properties
        self.n_samples = n_samples
        self.sample_size = sample_size
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
        return [text[i] for i in range(len(text))]
    # end convert

# end FunctionWord
