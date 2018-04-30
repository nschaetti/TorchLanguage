# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Set the input to a fixed length, truncate the tensor or extend it with zeros
class ToLength(Transformer):
    """
    Set the input to a fixed length, truncate the tensor or extend it with zeros
    """

    # Constructor
    def __init__(self, length, input_dim=0):
        """
        Constructor
        :param length: Fixed length
        """
        # Super constructor
        super(ToLength, self).__init__()

        # Properties
        self.length = length
        self.input_size = input_dim
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
        return self.input_size
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

        # For each sample
        if x.dim() > 0:
            for b in range(x.size(0)):
                if start:
                    result = self._transform(x[b]).unsqueeze(0)
                    start = False
                else:
                    result = torch.cat((result, self._transform(x[b]).unsqueeze(0)), dim=0)
                # end if
            # end for
        else:
            # Tensor type
            tensor_type = x.__class__
            if self.input_dim > 0:
                return tensor_type(1, self.length, x.size(1)).fill_(0)
            else:
                return tensor_type(1, self.length).fill_(0)
            # end if
        # end if

        return result
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
    # Private
    ##############################################

    # Transform
    def _transform(self, x):
        """
        Transform input
        :param x:
        :return:
        """
        # Tensor type
        tensor_type = x.__class__

        # Long Tensor (idx) or Float (embeddings)
        if type(x) == torch.FloatTensor or type(x) == torch.DoubleTensor:
            self.input_dim = x.size(1)
            new_tensor = tensor_type(self.length, x.size(1))
        else:
            new_tensor = tensor_type(self.length)
        # end if

        # Fill zero
        new_tensor.fill_(0)

        # Set
        if x.dim() == 0:
            return new_tensor
        else:
            if x.size(0) < self.length:
                new_tensor[:x.size(0)] = x
            else:
                new_tensor = x[:self.length]
            # end if
        # end if

        return new_tensor
    # end _transform

# end ToLength
