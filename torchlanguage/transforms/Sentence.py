# -*- coding: utf-8 -*-
#

# Imports
import spacy
from .Transformer import Transformer


# Transform text to a list of sentences
class Sentence(Transformer):
    """
    Transform text to a list of tokens
    """

    # Constructor
    def __init__(self, model="en_core_web_lg", lang='en'):
        """
        Constructor
        :param model: Spacy's model to load.
        """
        # Upper
        super(Sentence, self).__init__()

        # Properties
        self.lang = lang
        self.model = model
        self.nlp = spacy.load(model)
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
        # Inputs as a list
        sentences = list()

        # For each tokens
        for sentence in self.nlp(text).sents:
            sentences.append(unicode(sentence.text))
        # end for

        return sentences
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
