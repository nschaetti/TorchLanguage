# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer
import numpy as np


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

    @property
    def input_dim(self):
        """
        Input dimension
        :return:
        """
        return self.input_size
    # end input_dim

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, u):
        """
        Convert a string to a ESN input
        :param u: Signal to transform
        :return: Tensor or list
        """
        # Step
        if self.overlapse:
            step = 1
            last = 1
        else:
            step = self.n
            last = 0
        #  end if
        print(u)
        print(u.size())
        # List
        if type(u) is list:
            return [u[i:i+self.n] for i in np.arange(0, len(u) - last, step)]
        elif type(u) is torch.LongTensor or type(u) is torch.FloatTensor or type(u) is torch.Tensor:
            dtype = type(u)
            if u.dim() == 2:
                n_gram_tensor = dtype(u.size(0), self.n)
                for i in np.arange(0, len(u) - last, step):
                    n_gram_tensor[i] = u[i:i+self.n]
                # end for
            elif u.dim() == 3:
                n_gram_tensor = dtype(u.size(0), self.n, u.size(1))
                for i in np.arange(0, len(u) - last, step):
                    n_gram_tensor[i] = u[i:i+self.n]
                # end for
            # end if
            return n_gram_tensor
        # end if
    # end convert

# end ToNGram
