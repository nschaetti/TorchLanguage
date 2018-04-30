# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Replace index greater than max_id
class MaxIndex(Transformer):
    """
    Replace index greater than max_id
    """

    # Constructor
    def __init__(self, max_id, replace_by=0):
        """
        Constructor
        :param max_id: Index greater than this value are replaces by replace_by
        :param replace_by: New value for index greater than max_id
        """
        # Super constructor
        super(MaxIndex, self).__init__()

        # Properties
        self.max_id = max_id
        self.replace_by = replace_by
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
    def __call__(self, idx):
        """
        Convert a string to a ESN input
        :param text: Text to convert
        :return: Tensor of word vectors
        """
        # Replace
        idx[torch.gt(idx, self.max_id)] = self.replace_by
        return idx
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
        return 1
    # end if

    ##############################################
    # Static
    ##############################################

# end MaxIndex
