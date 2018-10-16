# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer
import numpy as np
import math


# Input signal n-gram
class ToNGram(Transformer):
    """
    Input signal to n-gram
    """

    # Constructor
    def __init__(self, n, overlapse=False):
        """
        Constructor
        """
        # Super constructor
        super(ToNGram, self).__init__()

        # Properties
        self.n = n
        self.overlapse = overlapse
        self.input_size = 1
    # end __init__

    ##############################################
    # Public
    ##############################################

    ##############################################
    # Properties
    ##############################################

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, x):
        """
        Convert a string to a ESN input
        :param x: Signal to transform
        :return: Tensor or list
        """
        # Add dim if needed
        if type(x) is list:
            return self._transform(x)
        elif type(x) is torch.LongTensor or type(x) is torch.FloatTensor or type(x) is torch.DoubleTensor or type(x) is torch.Tensor:
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
                return x
            # end if

            return result
        else:
            raise NotImplementedError(u"ToNGram for type other than list or tensor not implemented")
        # end if
    # end convert

    ##############################################
    # Private
    ##############################################

    # Transform
    def _transform(self, u):
        """
        Transform input
        :param x:
        :return:
        """
        # Step
        if self.overlapse:
            step = 1
            last = self.n
        else:
            step = self.n
            last = self.n - 1
        #  end if

        # List
        if type(u) is list:
            return [u[i:i + self.n] for i in np.arange(0, len(u) - last, step)]
        elif type(u) is torch.LongTensor or type(u) is torch.FloatTensor or type(u) is torch.Tensor or type(u) is torch.DoubleTensor:
            # Output type
            if "LongTensor" in u.type():
                dtype = torch.LongTensor
            elif "DoubleTensor" in u.type():
                dtype = torch.DoubleTensor
            else:
                dtype= torch.FloatTensor
            # end if

            # Length
            if self.overlapse:
                length = u.size(0) - self.n + 1
            else:
                length = int(math.floor(u.size(0) / self.n))
            # end if

            # Dimension
            if u.dim() == 1:
                n_gram_tensor = dtype(length, self.n).fill_(0)
                for i, j in enumerate(np.arange(0, u.size(0) - last, step)):
                    n_gram_tensor[i] = u[j:j + self.n]
                # end for
            elif u.dim() == 2:
                n_gram_tensor = dtype(length, self.n, u.size(1)).fill_(0)
                for i, j in enumerate(np.arange(0, u.size(0) - last, step)):
                    n_gram_tensor[i] = u[j:j + self.n]
                # end for
            # end if
            return n_gram_tensor
        # end if
    # end _transform

# end ToNGram
