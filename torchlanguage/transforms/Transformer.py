# -*- coding: utf-8 -*-
#

# Imports


# Base class for text transformers
class Transformer(object):
    """
    Base class for text transformers
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        pass
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
        pass
    # end input_dim

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, tokens):
        """
        Convert a string to a ESN input
        :param tokens: Text to convert
        :return: A list of symbols
        """
        pass
    # end convert

    # String
    def __str__(self):
        """
        String
        :return:
        """
        # Class
        init_str = type(self).__name__ + "("

        # For each attributes
        index = 0
        for attr in dir(self):
            if u"_" not in attr:
                attr_value = getattr(self, attr)
                if type(attr_value) is int or type(attr_value) is float or type(attr_value) is str or type(
                        attr_value) is unicode or type(attr_value) is tuple:
                    add_begin = " " if index != 0 else ""
                    init_str += add_begin + "{}={}, ".format(attr, getattr(self, attr))
                    index += 1
                # end if
            # end if
        # end for

        # Remove ", "
        if init_str[-2:] == ", ":
            init_str = init_str[:-2]
        # end if

        # )
        init_str += ")"

        return init_str
    # end __str__

    # end __str__

    # Unicode
    def __unicode__(self):
        """
        Unicode
        :return:
        """
        # Class
        init_str = unicode(type(self).__name__) + u"("

        # For each attributes
        index = 0
        for attr in dir(self):
            if "_" not in attr:
                attr_value = getattr(self, attr)
                if type(attr_value) is int or type(attr_value) is float or type(attr_value) is str or type(attr_value) is unicode or type(attr_value) is tuple:
                    add_begin = u" " if index != 0 else u""
                    init_str += add_begin + u"{}={}, ".format(attr, getattr(self, attr))
                    index += 1
                # end if
            # end if
        # end for

        # Remove ", "
        if init_str[-2:] == u", ":
            init_str = init_str[:-2]
        # end if

        # )
        init_str += u")"

        return init_str
    # end __unicode__

    ##############################################
    # Static
    ##############################################

    ##############################################
    # Private
    ##############################################

    # Transform
    def _transform(self, x):
        """
        Transform input
        :param x:
        :return:
        """
        pass
    # end _transform

# end TextTransformer
