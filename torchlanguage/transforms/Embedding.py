# -*- coding: utf-8 -*-
#

# Imports
import gensim
import torch
import torch.nn as nn
import numpy as np
from .Transformer import Transformer


# Transform text to vectors with embedding
class Embedding(Transformer):
    """
    Transform text to vectors with embedding
    """

    # Constructor
    def __init__(self, weights, voc_size):
        """
        Constructor
        :param weights: Embedding weight matrix
        """
        # Super constructor
        super(Embedding, self).__init__()

        # Properties
        self.weights = weights
        self.voc_size = voc_size
        self.embedding_dim = weights.size(1)
    # end __init__

    ##############################################
    # Properties
    ##############################################

    # Get the number of inputs
    @property
    def input_dim(self):
        """
        Get the number of inputs
        :return:
        """
        return self.embedding_dim
    # end input_dim

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, idxs):
        """
        Convert a strng
        :param idxs: Tensor of indexes
        :return:
        """
        # Start
        start = True

        # Result
        result = None

        # For each sample
        if idxs.dim() > 0:
            for b in range(idxs.size(0)):
                if start:
                    result = self._transform(idxs[b])
                    start = False
                else:
                    result = torch.cat((result, self._transform(idxs[b]).unsqueeze(0)), dim=0)
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
        :param idxs:
        :return:
        """
        # Inputs as tensor
        inputs = torch.FloatTensor()

        # Start
        start = True
        count = 0.0

        # OOV
        zero = 0.0
        self.oov = 0.0

        # For each inputs
        if idxs.dim() == 1:
            for i in range(idxs.size(0)):
                # Get token ix
                ix = idxs[i]

                # Get vector
                if ix < self.voc_size:
                    embedding_vector = self.weights[ix]
                else:
                    embedding_vector = torch.zeros(self.embedding_dim)
                # end if

                # Test zero
                if torch.sum(embedding_vector) == 0.0:
                    zero += 1.0
                    embedding_vector = torch.zeros(self.input_dim)
                # end if

                # Start/continue
                if not start:
                    inputs = torch.cat((inputs, embedding_vector.unsqueeze(0)), dim=0)
                else:
                    inputs = embedding_vector.unsqueeze(0)
                    start = False
                # end if
                count += 1
            # end for
        elif idxs.dim() == 2:
            for i in range(idxs.size(0)):
                embedding_inputs = torch.zeros(idxs.size(1), self.embedding_dim)
                for j in range(idxs.size(1)):
                    # Get token ix
                    ix = idxs[i, j]

                    # Get vector
                    if ix < self.voc_size:
                        embedding_vector = self.weights[ix]
                    else:
                        embedding_vector = torch.zeros(self.embedding_dim)
                    # end if

                    # Test zero
                    if torch.sum(embedding_vector) == 0.0:
                        zero += 1.0
                        embedding_vector = np.zeros(self.input_dim)
                    # end if

                    # Set
                    embedding_inputs[j] = embedding_vector
                    count += 1
                # end for
                if not start:
                    inputs = torch.cat((inputs, embedding_inputs.unsqueeze(0)), dim=0)
                else:
                    inputs = embedding_inputs.unsqueeze(0)
                    start = False
                # end if
            # end for
        else:
            raise NotImplementedError(u"Invalid tensor dimension : {}".format(idxs.size()))
        # end if

        # OOV
        self.oov = zero / count * 100.0

        return inputs.unsqueeze(0)
    # end _transform


# end Embedding
