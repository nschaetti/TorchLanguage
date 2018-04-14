# -*- coding: utf-8 -*-
#

# Imports
import torch


# Replace an index or a vector by zero or a given value with a given probability
class DropOut(object):
    """
    Replace an index or a vector by zero or a given value with a given probability
    """

    # Constructor
    def __init__(self, prob, replace_by=0):
        """
        Constructor
        :param prob:
        """
        # Properties
        self.prob = prob
        self.replace_by = replace_by
    # end __init__

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, x):
        """
        Convert a string to a ESN input
        :param x: Tensor
        :return: Tensor of word vectors
        """
        # Tensor type
        tensor_type = x.__class__

        # Mask
        mask = torch.bernoulli(torch.FloatTensor(x.size(0)).fill_(1.0 - self.prob))

        # Empty tensor
        if x.dim() == 2:
            self.input_dim = x.size(1)
            replace_tensor = tensor_type(self.input_dim)
            replace_tensor.fill_(self.replace_by)
            for i in range(x.size(0)):
                if mask[i] == 0:
                    x[i] = replace_tensor
                # end if
            # end for
        else:
            replace_tensor = self.replace_by
            x[torch.eq(mask, 0)] = replace_tensor
        # end if

        return x
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

# end DropOut
