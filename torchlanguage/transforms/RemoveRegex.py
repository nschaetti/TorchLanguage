# -*- coding: utf-8 -*-
#

# Imports
import re
from .Transformer import Transformer


# Remove regex in the text
class RemoveRegex(Transformer):
    """
    Remove regex in the text
    """

    # Constructor
    def __init__(self, regex):
        """
        Constructor
        """
        # Super constructor
        super(RemoveRegex, self).__init__()

        # Properties
        self.regex = regex
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
        return re.sub(self.regex, '', text)
    # end convert

# end RemoveRegex
