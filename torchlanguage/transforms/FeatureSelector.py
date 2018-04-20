# -*- coding: utf-8 -*-
#

# Imports
import echotorch.nn


# Transform input vectors with feature selector
class FeatureSelector(object):
    """
    Transform input vectors with feature selector
    """

    # Constructor
    def __init__(self, model, remove_last_layer=True):
        """
        Constructor
        :param model: Feature selection model.
        """
        # Properties
        self.model = model
        self.input_dim = self.model.fc.out_features

        # Remove last layer
        if remove_last_layer:
            self.model.fc = echotorch.nn.Identity()
        # end if
    # end __init__

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
