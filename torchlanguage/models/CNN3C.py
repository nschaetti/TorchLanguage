# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# CNN with 3 channels filters
class CNN3C(nn.Module):
    """
    CNN with 3 channels filters
    """

    # Constructor
    def __init__(self, voc_size, embedding_dim, n_authors, window_size, out_channels=(500, 500, 500), kernel_sizes=(3, 4, 5), temporal_max=300, n_features=30):
        """
        Constructor
        :param n_authors:
        :param out_channels: Number of output channels
        :param kernel_sizes: Different kernel sizes
        """
        super(CNN3C, self).__init__()
        self.n_features = n_features
        self.n_authors = n_authors

        # Embedding layer
        self.embedding = nn.Embedding(voc_size, embedding_dim)

        # Conv window 1
        self.conv_w1 = nn.Conv2d(in_channels=1, out_channels=out_channels[0],
                                 kernel_size=(kernel_sizes[0], embedding_dim))

        # Conv window 2
        self.conv_w2 = nn.Conv2d(in_channels=1, out_channels=out_channels[1],
                                 kernel_size=(kernel_sizes[1], embedding_dim))

        # Conv window 3
        self.conv_w3 = nn.Conv2d(in_channels=1, out_channels=out_channels[2],
                                 kernel_size=(kernel_sizes[2], embedding_dim))

        # Max pooling layer
        self.max_pool_w1 = nn.MaxPool1d(kernel_size=temporal_max, stride=temporal_max)
        self.max_pool_w2 = nn.MaxPool1d(kernel_size=temporal_max, stride=temporal_max)
        self.max_pool_w3 = nn.MaxPool1d(kernel_size=temporal_max, stride=temporal_max)

        # Linear layer 1
        # self.linear_size = out_channels[1] * 72
        self.temporal_features = int(math.floor(window_size / temporal_max))
        self.linear_size = self.temporal_features * out_channels[0] + self.temporal_features * out_channels[
            1] + self.temporal_features * out_channels[2]
        self.linear = nn.Linear(self.linear_size, n_features)

        # Linear layer 2
        self.linear2 = nn.Linear(self.n_features, n_authors)
    # end __init__

    # Forward
    def forward(self, x):
        """
        Forward
        :param x:
        :return:
        """
        # Embedding layer
        embeds = self.embedding(x)

        # Add channel dim
        embeds = torch.unsqueeze(embeds, dim=1)

        # Conv window
        out_win1 = F.relu(self.conv_w1(embeds))
        out_win2 = F.relu(self.conv_w2(embeds))
        out_win3 = F.relu(self.conv_w3(embeds))

        # Remove last dim
        out_win1 = torch.squeeze(out_win1, dim=3)
        out_win2 = torch.squeeze(out_win2, dim=3)
        out_win3 = torch.squeeze(out_win3, dim=3)

        # Max pooling
        max_win1 = self.max_pool_w1(out_win1)
        max_win2 = self.max_pool_w2(out_win2)
        max_win3 = self.max_pool_w3(out_win3)

        # Concatenate
        out = torch.cat((max_win1, max_win2, max_win3), dim=1)

        # Flatten
        out = out.view(-1, self.linear_size)

        # Linear 1
        out = F.relu(self.linear(out))

        # Linear 2
        out = F.relu(self.linear2(out))

        # Log softmax
        log_prob = F.log_softmax(out, dim=1)

        # Log Softmax
        return log_prob
    # end forward

# end CNN3C
