# -*- coding: utf-8 -*-
#

# Imports
import torch
from .Transformer import Transformer


# Transform tokens to index
class ToIndex(Transformer):
    """
    Transform tokens to index
    """

    # Constructor
    def __init__(self, token_to_ix=None, start_ix=0):
        """
        Constructor
        :param model: Spacy's model to load.
        """
        # Super constructor
        super(ToIndex, self).__init__()

        # Gram to ix
        if token_to_ix is not None:
            self.token_count = len(token_to_ix.keys())
            self.token_to_ix = token_to_ix
        else:
            self.token_count = start_ix
            self.token_to_ix = dict()
        # end if

        # Ix to gram
        self.ix_to_token = dict()
        if token_to_ix is not None:
            for token in token_to_ix.keys():
                self.ix_to_token[token_to_ix[token]] = token
            # end for
        # end if
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

    # Vocabulary size
    @property
    def voc_size(self):
        """
        Vocabulary size
        :return:
        """
        return self.token_count
    # end voc_size

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
        # Add to voc
        for i in range(len(text)):
            token = text[i]
            # if token not in self.token_to_ix.keys():
            try:
                ix = self.token_to_ix[token]
            except KeyError:
                self.token_to_ix[token] = self.token_count
                self.ix_to_token[self.token_count] = token
                self.token_count += 1
            # end if
        # end for

        # List of character to 2grams
        text_idxs = [self.token_to_ix[text[i]] for i in range(len(text))]

        # To long tensor
        text_idxs = torch.LongTensor(text_idxs)
        if text_idxs.dim() > 0:
            text_idxs = text_idxs.unsqueeze(0)
        # end if

        return text_idxs
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

# end ToIndex
