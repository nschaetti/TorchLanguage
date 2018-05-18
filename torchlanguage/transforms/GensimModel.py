# -*- coding: utf-8 -*-
#

# Imports
import gensim
import torch
import numpy as np
from .Transformer import Transformer


# Transform text to vectors with a Gensim model
class GensimModel(Transformer):
    """
    Transform text to vectors with a Gensim model
    """

    # Constructor
    def __init__(self, model_path):
        """
        Constructor
        :param model_path: Model's path.
        """
        # Super constructor
        super(GensimModel, self).__init__()

        # Properties
        self.model_path = model_path

        # Format
        binary = False if model_path[-4:] == ".vec" else True

        # Load
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=binary, unicode_errors='ignore')

        # OOV
        self.oov = 0.0
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
        return 300
    # end input_dim

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, tokens):
        """
        Convert a string to a ESN input
        :param text: Text to convert
        :return: Tensor of word vectors
        """
        return self._transform(tokens).unsqueeze(0)
    # end convert

    ##############################################
    # Private
    ##############################################

    # Transform
    def _transform(self, tokens):
        """
        Transform input
        :param tokens:
        :return:
        """
        # Text length
        text_length = len(tokens)

        # Inputs as tensor
        inputs = torch.zeros(text_length, self.input_dim)

        # Start
        count = 0.0

        # OOV
        zero = 0.0
        self.oov = 0.0

        # For each tokens
        for index, token in enumerate(tokens):
            found = False
            # Try normal
            try:
                word_vector = self.model[token]
                found = True
            except KeyError:
                pass
            # end try

            # Try lower
            if not found:
                try:
                    word_vector = self.model[token.lower()]
                except KeyError:
                    zero += 1.0
                    word_vector = np.zeros(self.input_dim)
                # end try
            # end if

            # Start/continue
            inputs[index] = torch.from_numpy(word_vector)
            count += 1.0
        # end for

        # OOV
        self.oov = zero / count * 100.0

        return inputs
    # end _transform

# end GensimModel
