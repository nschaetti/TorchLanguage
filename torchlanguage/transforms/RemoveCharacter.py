# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Remove a set of character in the text
class RemoveCharacter(Transformer):
    """
    Remove a set of character in the text
    """

    # Constructor
    def __init__(self, char_to_remove):
        """
        Constructor
        """
        # Super constructor
        super(RemoveCharacter, self).__init__()
        self.char_to_remove = char_to_remove
    # end __init__

    ##############################################
    # Public
    ##############################################

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
        # Remove chars
        for c in self.char_to_remove:
            text = text.replace(c, u"")
        # end for
        return text
    # end convert

# end FunctionWord
