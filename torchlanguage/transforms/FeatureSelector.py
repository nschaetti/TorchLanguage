# -*- coding: utf-8 -*-
#

# Imports
from torch.autograd import Variable
from .Transformer import Transformer


# Transform input vectors with feature selector
class FeatureSelector(Transformer):
    """
    Transform input vectors with feature selector
    """

    # Constructor
    def __init__(self, model, n_features, to_variable=False):
        """
        Constructor
        :param model: Feature selection model.
        """
        # Super constructor
        super(FeatureSelector, self).__init__()

        # Properties
        self.model = model
        self.input_size = n_features
        self.to_variable = to_variable
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
        return self.input_size
    # end input_dim

    ##############################################
    # Override
    ##############################################

    # Convert a string
    def __call__(self, x):
        """
        Convert a string to a ESN input
        :param x: Tensor to transform
        :return: Tensor of word vectors
        """
        return self._transform(x)
    # end convert

    ##############################################
    # Private
    ##############################################

    # Transform
    def _transform(self, x):
        """
        Transform input
        :param text:
        :return:
        """
        if self.to_variable:
            transformed = self.model(Variable(x))
            return transformed.data
        else:
            return self.model(x)
        # end if
    # end _transform

# end FeatureSelector
