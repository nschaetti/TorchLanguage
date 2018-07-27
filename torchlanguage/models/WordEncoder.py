# -*- coding: utf-8 -*-
#

# Imports
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['wEnc']


# Word encodger
class WordEncoder(nn.Module):
    """
    Word encoder
    """

    # Constructor
    def __init__(self, n_gram=3, n_features=(600, 300), embedding_size=300):
        """
        Consturctor
        :param n_classes: Number of output classes
        :param n_gram: How many consecutive words as input
        :param n_features: Number of hidden features
        :param embedding_size: Input embedding dimension
        """
        super(WordEncoder, self).__init__()
        self.n_features = n_features
        self.inputs_size = n_gram * embedding_size
        self.encode = False

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.inputs_size, n_features[0]),
            nn.ReLU(True),
            nn.Linear(n_features[0], n_features[1])
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_features[1], n_features[0]),
            nn.ReLU(True),
            nn.Linear(n_features[0], self.inputs_size),
            nn.Tanh()
        )
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

        # Encoder
        out = self.encoder(out)

        # Normalize
        out = F.normalize(out, p=2, dim=1)

        # Decoder if in training
        if not self.encode:
            out = self.decoder(out)
        # end if

        # Outputs
        return out
    # end forward

# end WordEncoder
