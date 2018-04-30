# -*- coding: utf-8 -*-
#

# Imports
import spacy
import nltk
from .Transformer import Transformer


# Transform text to a list of tokens
class Token(Transformer):
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
        super(Token, self).__init__()

        # Properties
        self.lang = lang
        self.model = model
        if model != 'nltk':
            self.nlp = spacy.load(model)
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
        tokens = list()

        # For each tokens
        if self.model == 'nltk':
            for token in nltk.word_tokenize(text, language=self.lang):
                tokens.append(unicode(token))
            # end for
        else:
            for token in self.nlp(text):
                tokens.append(unicode(token.text))
            # end for
        # end if

        return tokens
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
