# -*- coding: utf-8 -*-
#

# Imports
import torch


# Set the input to a fixed length, truncate the tensor or extend it with zeros
class ToLength(object):
    """
    Set the input to a fixed length, truncate the tensor or extend it with zeros
    """

    # Constructor
    def __init__(self, length):
        """
        Constructor
        :param length: Fixed length
        """
        # Properties
        self.length = length
        self.input_dim = 1
    # end __init__

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, x):
        """
        Convert a string to a ESN input
        :param text: Text to convert
        :return: Tensor of word vectors
        """
        # Tensor type
        tensor_type = x.__class__

        # Empty tensor
        if x.dim() == 2:
            self.input_dim = x.size(1)
            new_tensor = tensor_type(self.length, x.size(1))
        else:
            new_tensor = tensor_type(self.length)
        # end if

        # Fill zero
        new_tensor.fill_(0)

        # Set
        if x.size(0) < self.length:
            new_tensor[:x.size(0)] = x
        else:
            new_tensor = x[:self.length]
        # end if

        return new_tensor
    # end convert

    ##############################################
    # Private
    ##############################################

    # Get inputs size
    def _get_inputs_size(self):
        """
        Get inputs size.
        :return:
        """
        return self.input_dim
    # end if

    ##############################################
    # Static
    ##############################################

# end ToLength
