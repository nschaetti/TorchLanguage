# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Transform idxs tensor to frequency vector
class ToFrequencyVector(Transformer):
    """
    Transform idxs tensor to frequency vector
    """

    # Constructor
    def __init__(self, voc_size):
        """
        Constructor
        :param voc_size: Number of tokens in the vocabulary
        """
        # Super constructor
        super(ToFrequencyVector, self).__init__()

        # Properties
        self.voc_size = voc_size
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
        return self.voc_size
    # end input_dim

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, idxs):
        """
        Convert a string to a ESN input
        :param idxs: Tensor of indexes
        :return: Tensor of frequencies
        """
        # Start
        start = True

        # Result
        result = None

        # For each sample
        if idxs.dim() > 0:
            for b in range(idxs.size(0)):
                freq_tensor = self._transform(idxs[b])
                freq_tensor = freq_tensor.unsqueeze(0)
                if start:
                    result = freq_tensor
                    start = False
                else:
                    result = torch.cat((result, freq_tensor), dim=0)
                # end if
            # end for
        else:
            return idxs
        # end if

        return result
    # end convert

    ##############################################
    # Private
    ##############################################

    # Transform
    def _transform(self, idxs):
        """
        Transform input
        :param x:
        :return:
        """
        # Tensor with zeros
        freq_tensor = torch.zeros(self.voc_size)

        # Add each idxs
        for i in range(idxs.size(0)):
            freq_tensor[idxs[i]] += 1.0
        # end for

        # Frequency
        freq_tensor /= idxs.size(0)

        return freq_tensor
    # end _transform

    # Get inputs size
    def _get_inputs_size(self):
        """
        Get inputs size.
        :return:
        """
        return self.voc_size
    # end if

    ##############################################
    # Static
    ##############################################

# end ToFrequencyVector
