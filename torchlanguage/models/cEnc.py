# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['cEnc']


# Character encoder
class cEnc(nn.Module):
    """
    Character encoder
    """

    # Constructor
    def __init__(self, text_length, vocab_size, n_classes, embedding_dim=50, n_features=300):
        """
        Constructor
        :param text_length: Text's length
        :param vocab_size: Vocabulary size
        :param n_classes: Number of output classes
        :param embedding_dim: Embedding size
        :param n_features: Number of hidden features
        """
        super(cEnc, self).__init__()

        # Properties
        self.embedding_dim = embedding_dim
        self.text_length = text_length
        self.n_classes = n_classes
        self.n_features = n_features

        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Linear layer
        self.inputs_size = text_length * embedding_dim
        self.linear = nn.Linear(self.inputs_size, self.n_features)
        self.linear2 = nn.Linear(self.n_features, self.n_classes)
    # end __init__

    # Forward
    def forward(self, x):
        """
        Forward
        :param x:
        :return:
        """
        # Embeddings
        embeds = self.embeddings(x)

        # Flatten
        out = embeds.view(-1, self.inputs_size)

        # Linear
        out = F.sigmoid(self.linear(out))

        # Linear 2
        out = self.linear2(out)

        # Outputs
        return out
    # end forward

# end cEnc
