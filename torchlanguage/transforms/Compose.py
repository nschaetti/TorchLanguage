# -*- coding: utf-8 -*-
#

# Imports
from .Transformer import Transformer


# Compose multiple transformations
class Compose(Transformer):
    """
    Compose multiple transformations
    """

    # Constructor
    def __init__(self, transforms):
        """
        Constructor
        """
        # Properties
        self.transforms = transforms

        # Super constructor
        super(Compose, self).__init__()
    # end __init__

    ##############################################
    # Public
    ##############################################

    ##############################################
    # Private
    ##############################################

    # Get to str for sub transforms
    def _transforms_str(self):
        """
        Get to str for sub transforms
        :return:
        """
        # String
        final_str = ""

        # For each transforms
        for trans in self.transforms:
            # Trans string
            tran_str = str(trans)

            # Add tab
            tran_str = "\t" + tran_str

            # Add tab to lines
            tran_str = tran_str.replace("\n", "\n\t")

            # Add
            final_str += tran_str + "\n"
        # end for

        # Remove final \n
        if final_str[-1] == "\n":
            final_str = final_str[:-1]
        # end if

        return final_str
    # end _transforms_str

    # Get to unicode for sub transforms
    def _transforms_unicode(self):
        """
        Get to unicode for sub transforms
        :return:
        """
        # String
        final_str = u""

        # For each transforms
        for trans in self.transforms:
            # Trans string
            tran_str = unicode(trans)

            # Add tab
            tran_str = u"\t" + tran_str

            # Add tab to lines
            tran_str = tran_str.replace(u"\n", u"\n\t")

            # Add
            final_str += tran_str + u"\n"
        # end for

        # Remove final \n
        if final_str[-1] == u"\n":
            final_str = final_str[:-1]
        # end if

        return final_str
    # end _transforms_unicode

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
        return self.transforms[-1].input_dim
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
        # For each transform
        for index, transform in enumerate(self.transforms):
            # Transform
            if index == 0:
                outputs = transform(text)
            else:
                outputs = transform(outputs)
            # end if
        # end for

        return outputs
    # end convert

    # String
    def __str__(self):
        """
        String
        :return:
        """
        return "Compose (\n{}\n)".format(self._transforms_str())

    # end __str__

    # Unicode
    def __unicode__(self):
        """
        Unicode
        :return:
        """
        return u"Compose (\n{}\n)".format(self._transforms_unicode())
    # end __unicode__

# end Compose
