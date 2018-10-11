# -*- coding: utf-8 -*-
#

# Imports
import spacy
import nltk
from .Transformer import Transformer


# Print representations
class Print(Transformer):
    """
    Print representations
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        # Upper
        super(Print, self).__init__()
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
    def __call__(self, text):
        """
        Convert a string to a ESN input
        :param text: Text to convert
        :return: Tensor of word vectors
        """
        print(text)

        return text
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

# end Token
