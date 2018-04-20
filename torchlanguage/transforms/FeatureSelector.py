# -*- coding: utf-8 -*-
#

# Imports


# Transform input vectors with feature selector
class FeatureSelector(object):
    """
    Transform input vectors with feature selector
    """

    # Constructor
    def __init__(self, model):
        """
        Constructor
        :param model: Feature selection model.
        """
        # Properties
        self.model = model
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
        return self.model.fc.out_features
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
        return self.model(x)
    # end _transform

# end FeatureSelector
