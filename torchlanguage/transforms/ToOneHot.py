# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Transform index to one-hot vector
class ToOneHot(Transformer):
    """
    Transform tokens to one-hot vector
    """

    # Constructor
    def __init__(self, voc_size):
        """
        Constructor
        :param voc_size: Vocabulary size
        """
        # Super constructor
        super(ToOneHot, self).__init__()

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
    def __call__(self, x):
        """
        Convert a string to a ESN input
        :param idxs: Indexes to convert
        :return: Tensor of word vectors
        """
        # Start
        start = True

        # Result
        result = None

        # For each sample
        if x.dim() > 0:
            for b in range(x.size(0)):
                transformed = self._transform(x[b])
                if start:
                    result = transformed
                    start = False
                else:
                    result = torch.cat((result, transformed), dim=0)
                # end if
            # end for
        else:
            return x
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
        return self.voc_size
    # end if

    # Transform
    def _transform(self, idxs):
        """
        Transform input
        :param x:
        :return:
        """
        # Inputs as tensor
        inputs = torch.FloatTensor()

        # Start
        start = True

        # For each tokens
        if idxs.dim() > 0:
            for i in range(idxs.size(0)):
                # One hot vector
                one_hot = torch.zeros(1, self.input_dim)
                one_hot[0, idxs[i]] = 1.0

                if not start:
                    inputs = torch.cat((inputs, one_hot), dim=0)
                else:
                    inputs = one_hot
                    start = False
                # end if
            # end for
            return inputs.unsqueeze(0)
        else:
            return inputs
        # end if
    # end _transform

    ##############################################
    # Static
    ##############################################

# end ToOneHot
