# -*- coding: utf-8 -*-
#

# Imports
import torch
import spacy
from .Transformer import Transformer


# Transform text to tag vectors
class Tag(Transformer):
    """
    Transform text to tag vectors
    """

    # Constructor
    def __init__(self, model="en_core_web_lg"):
        """
        Constructor
        :param model: Spacy's model to load.
        """
        # Super constructor
        super(Tag, self).__init__()

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
        Get all tags.
        :return: A list of tags.
        """
        return [u"''", u",", u":", u".", u"``", u"-LRB-", u"-RRB-", u"AFX", u"CC", u"CD", u"DT", u"EX", u"FW",
                u"IN", u"JJ", u"JJR", u"JJS", u"LS", u"MD", u"NN", u"NNS", u"NNP", u"NNPS", u"PDT", u"POS", u"PRP",
                u"PRP$", u"RB", u"RBR", u"RBS", u"RP", u"SYM", u"TO", u"UH", u"VB", u"VBZ", u"VBP", u"VBD", u"VBN",
                u"VBG", u"WDT", u"WP", u"WP$", u"WRB", u"X"]
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
        # List of tags
        tag_list = list()

        # For each tokens
        for token in self.nlp(text):
            # Replace if not function word
            if token.tag_ not in self.get_tags():
                token_tag = u"X"
            else:
                token_tag = token.tag_
            # end if

            # Add
            tag_list.append(token_tag)
        # end for

        return tag_list
    # end convert

# end FunctionWord
