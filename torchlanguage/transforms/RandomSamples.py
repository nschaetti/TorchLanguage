# -*- coding: utf-8 -*-
#

# Imports
import torch
import numpy as np
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
    def __call__(self, x):
        """
        Convert a string to a ESN input
        :param x: Input tensor
        :return: Tensor of word vectors
        """
        # Start
        start = True

        # Result
        result = None

        # Empty tensor
        if x.dim() == 0:
            return x
        # end if

        # Length
        if type(x) is torch.LongTensor:
            length = x.size(1)
        elif type(x) is torch.FloatTensor and type(x) is torch.DoubleTensor:
            length = x.size(2)
        # end if

        # For each sample
        for n in range(self.n_samples):
            start_index = np.random.randint(0, length - self.sample_size)
            sample = x[:, start_index:start_index + self.sample_size]
            if start:
                result = sample
                start = False
            else:
                result = torch.cat((result, sample), dim=0)
            # end if
        # end for

        return result
    # end convert

# end FunctionWord
