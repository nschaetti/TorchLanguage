# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['cEnc']


# Character encoder
class CharacterEncoder(nn.Module):
    """
    Character encoder
    """

    # Constructor
    def __init__(self, text_length, vocab_size, embedding_dim=50, n_features=(600, 300)):
        """
        Constructor
        :param text_length: Text's length
        :param vocab_size: Vocabulary size
        :param n_classes: Number of output classes
        :param embedding_dim: Embedding size
        :param n_features: Number of hidden features
        """
        super(CharacterEncoder, self).__init__()

        # Properties
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.text_length = text_length
        self.n_features = n_features
        self.output_dim = text_length * vocab_size
        self.inputs_size = text_length * embedding_dim
        self.encode = False

        # Embedding
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.inputs_size, self.n_features[0]),
            nn.ReLU(True),
            nn.Linear(self.n_features[0], self.n_features[1])
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.n_features[1], self.n_features[0]),
            nn.ReLU(True),
            nn.Linear(self.n_features[0], self.output_dim),
        )
    # end __init__

    # Forward
    def forward(self, x):
        """
        Forward
        :param x:
        :return:
        """
        # Embedding
        x = self.embeddings(x)

        # Reshape
        x = x.view(-1, self.inputs_size)

        # Encoder
        x = self.encoder(x)

        # Normalize
        x = F.normalize(x, p=2, dim=1)

        # Decode if in training
        if not self.encode:
            # Decoder
            x = self.decoder(x)

            # Reshape
            x = x.view(-1, self.text_length, self.vocab_size)
        # end if

        return x
    # end forward

# end CharacterEncoder
