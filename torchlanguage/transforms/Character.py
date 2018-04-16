#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : torchlanguage.transforms.Character
# Description : Transform text to list of character.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing NLP Project.
# The Reservoir Computing Memory Project is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

# Imports
from .Transformer import Transformer


# Transform text to character vectors
class Character(Transformer):
    """
    Transform text to character vectors
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        # Super constructor
        super(Character, self).__init__()
    # end __init__

    ##############################################
    # Public
    ##############################################

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
        # List of character
        return self._transform(text)
    # end convert

    ##############################################
    # Private
    ##############################################

    # Transform
    def _transform(self, x):
        """
        Transform
        :param x:
        :return:
        """
        return [x[i] for i in range(len(x))]
    # end _transform

# end FunctionWord
