# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Take a list of transformer (to tensor) and stack them
# on the time dimension.
class HorizontalStack(Transformer):
    """
    Take a list of transformer (to tensor) and stack them
    on the time dimension.
    """

    # Constructor
    def __init__(self, transformers):
        """
        Constructor
        """
        # Super constructor
        super(HorizontalStack, self).__init__()

        # Properties
        self.transformers = transformers
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

        # For each transformers
        for t in self.transformers:
            if start:
                result = t(x)
                start = False
            else:
                result = torch.cat((result, t(x)), dim=1)
            # end if
        # end for

        return result
    # end convert

# end HorizontalStack
