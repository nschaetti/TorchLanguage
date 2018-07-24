# -*- coding: utf-8 -*-
#

# Imports
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['wEnc']


# Word encodger
class wEnc(nn.Module):
    """
    Word encoder
    """

    # Constructor
    def __init__(self, n_classes, n_gram=3, n_features=300, embedding_size=300):
        """
        Consturctor
        :param n_classes: Number of output classes
        :param n_gram: How many consecutive words as input
        :param n_features: Number of hidden features
        :param embedding_size: Input embedding dimension
        """
        super(wEnc, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.inputs_size = n_gram * embedding_size

        # Linear layer 1
        self.linear = nn.Linear(self.inputs_size, n_features)

        # Linear layer 2
        self.linear2 = nn.Linear(n_features, n_classes)
    # end __init__

    # Forward
    def forward(self, x):
        """
        Forward
        :param x:
        :return:
        """
        # Flatten
        out = x.view(-1, self.inputs_size)

        # Linear 1
        out = F.sigmoid(self.linear(out))

        # Linear 2
        out = self.linear2(out)

        # Outputs
        return out
    # end forward

# end wEnc
