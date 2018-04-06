# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN gram
class CNNgram(nn.Module):
    """
    CNN gram
    """

    # Constructor
    def __init__(self, voc_size, embedding_dim, n_gram, n_classes=15, out_channels=(20, 10), kernel_sizes=(5, 5), max_pool_size=2, n_features=30):
        """
        Constructor
        :param n_authors:
        :param out_channels: Number of output channels
        :param kernel_sizes: Different kernel sizes
        """
        super(CNNgram, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_gram = n_gram

        # Embedding layer
        self.embedding = nn.Embedding(voc_size, embedding_dim)

        # Conv 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels[0], kernel_size=(n_gram, kernel_sizes[0]))

        # Max pooling layer
        self.max_pool = nn.MaxPool1d(kernel_size=max_pool_size, stride=0)

        # Conv 2
        # self.conv2 = nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_sizes[1])

        # Linear layer 1
        # self.linear_size = out_channels[1] * 72
        # self.linear_size = out_channels[0] * 148
        self.linear_size = out_channels[0] * 23
        self.linear = nn.Linear(self.linear_size, n_features)

        # Linear layer 2
        self.linear2 = nn.Linear(self.n_features, n_classes)
    # end __init__

    # Forward
    def forward(self, x):
        """
        Forward
        :param x:
        :return:
        """
        # Embedding layer
        embed = self.embedding(x)

        # Channel
        embed = embed.unsqueeze(1)

        # Conv 1
        out_conv1 = F.relu(self.conv1(embed))

        # Unsqueeze
        out_conv1 = out_conv1.squeeze(2)

        # Max pooling
        max_pooled = self.max_pool(out_conv1)

        # Conv 2
        # out_conv2 = F.relu(self.conv2(max_pooled))

        # Max pooling
        # max_pooled = self.max_pool(out_conv2)

        # Flatten
        out = max_pooled.view(-1, self.linear_size)

        # Linear 1
        out = F.relu(self.linear(out))

        # Linear 2
        out = F.relu(self.linear2(out))

        # Log softmax
        log_prob = F.log_softmax(out, dim=1)

        # Log Softmax
        return log_prob
    # end forward

# end CNNgram
