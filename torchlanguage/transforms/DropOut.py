# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Replace an index or a vector by zero or a given value with a given probability
class DropOut(Transformer):
    """
    Replace an index or a vector by zero or a given value with a given probability
    """

    # Constructor
    def __init__(self, prob, replace_by=0):
        """
        Constructor
        :param prob:
        """
        # Super constructor
        super(DropOut, self).__init__()

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
        # Start
        start = True

        # Result
        result = None

        # For each sample
        if x.dim() > 0:
            for b in range(x.size(0)):
                transformed = self._transform(x[b]).unsqueeze(0)
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

    # Transform
    def _transform(self, x):
        """
        Transform input
        :param x:
        :return:
        """
        # Tensor type
        tensor_type = x.__class__

        # Mask
        mask = torch.bernoulli(torch.FloatTensor(x.size(0)).fill_(1.0 - self.prob))

        # Empty tensor
        if type(x) is torch.FloatTensor or type(x) is torch.DoubleTensor:
            self.input_dim = x.size(1)
            replace_tensor = tensor_type(self.input_dim)
            replace_tensor.fill_(self.replace_by)
            for i in range(x.size(0)):
                if mask[i] == 0:
                    x[i] = replace_tensor
                # end if
            # end for
        elif type(x) is torch.LongTensor:
            replace_tensor = self.replace_by
            x[torch.eq(mask, 0)] = replace_tensor
        else:
            raise NotImplementedError(
                u"Transformation on type other than Float/DoubleTensor or LongTensor not implemented"
            )
        # end if

        return x
    # end _transform

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
