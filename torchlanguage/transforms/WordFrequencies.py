# -*- coding: utf-8 -*-
#

# Imports
import torch
import spacy
import numpy as np
from .Transformer import Transformer


# Make statistics about words
class WordFrequencies(Transformer):
    """
    Make statistics about words
    """

    # Constructor
    def __init__(self, model="en_vectors_web_lg"):
        """
        Constructor
        :param model: Spacy's model to load.
        """
        # Super constructor
        super(WordFrequencies, self).__init__()

        # Properties
        self.model = model
        self.nlp = spacy.load(model)
        self.token_count = dict()
        self.token_total = 0
        self.tokens = list()
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
        return self._transform(text)
    # end convert

    ##############################################
    # Private
    ##############################################

    # Transform
    def _transform(self, text):
        """
        Transform input
        :param text:
        :return:
        """
        # Statistics
        self.token_count = dict()
        self.token_total = 0.0
        self.tokens = list()

        # For each tokens
        for token in self.nlp(text):
            try:
                self.token_count[token.text] += 1.0
            except KeyError:
                self.token_count[token.text] = 1.0
            # end try
            self.token_total += 1.0
            self.tokens.append(token.text)
        # end for

        return text
    # end _transform

# end WordFrequencies
