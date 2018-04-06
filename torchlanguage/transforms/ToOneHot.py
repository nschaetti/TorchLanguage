# -*- coding: utf-8 -*-
#

# Imports
import torch


# Transform index to one-hot vector
class ToOneHot(object):
    """
    Transform tokens to one-hot vector
    """

    # Constructor
    def __init__(self, voc_size):
        """
        Constructor
        :param model: Spacy's model to load.
        """
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
        :param idxs: Indexes to convert
        :return: Tensor of word vectors
        """
        # Inputs as tensor
        inputs = torch.FloatTensor(1, self.input_dim)

        # Start
        start = True

        # For each tokens
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

        return inputs
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

    ##############################################
    # Static
    ##############################################

# end ToOneHot
