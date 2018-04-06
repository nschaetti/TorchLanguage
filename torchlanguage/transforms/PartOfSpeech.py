# -*- coding: utf-8 -*-
#

# Imports
import torch
import spacy
from .Transformer import Transformer


# Transform text to part-of-speech vectors
class PartOfSpeech(Transformer):
    """
    Transform text to part-of-speech vectors
    """

    # Constructor
    def __init__(self, model="en_core_web_lg"):
        """
        Constructor
        :param model: Spacy's model to load.
        """
        # Super constructor
        super(PartOfSpeech, self).__init__()

        # Properties
        self.model = model
        self.nlp = spacy.load(model)
    # end __init__

    ##############################################
    # Public
    ##############################################

    # Get tags
    def get_tags(self):
        """
        Get tags.
        :return: A list of tags.
        """
        return [u"ADJ", u"ADP", u"ADV", u"CCONJ", u"DET", u"INTJ", u"NOUN", u"NUM", u"PART", u"PRON", u"PROPN",
                u"PUNCT", u"SYM", u"VERB", u"SPACE", u"X"]
    # end get_tags

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
        # Pos
        pos_list = []

        # For each tokens
        for token in self.nlp(text):
            if token.pos_ in self.get_tags():
                pos_list.append(token.pos_)
            else:
                pos_list.append(u"X")
            # end if
        # end for

        return pos_list
    # end convert

# end PartOfSpeech
