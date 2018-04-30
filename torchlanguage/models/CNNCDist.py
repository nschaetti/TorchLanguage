# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['CNNCDist', 'cnncdist']


# Model URLs
model_urls = {
    'cnncdist': {
        'en': 'https://www.nilsschaetti.com/models/cnncdist-en-bd63e232.pth'
    }
}

# Voc URLs
voc_urls = {
    'cnncdist': {
        'en': 'https://www.nilsschaetti.com/models/cnncdist-voc-en-bd63e232.pth'
    }
}


# CNN on character for distance learning
class CNNCDist(nn.Module):
    """
    CNN on character for distance learning
    """

    # Constructor
    def __init__(self, window_size, vocab_size, embedding_dim=50, out_channels=(500, 500, 500),
                 kernel_sizes=(3, 4, 5), linear_size=1500):
        """
        Constructor
        :param vocab_size: Vocabulary size
        :param embedding_dim: Embedding layer dimension
        :param out_channels: Number of output channels
        :param kernel_sizes: Different kernel sizes
        """
        super(CNNCDist, self).__init__()

        # Properties
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.linear_size = linear_size

        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

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
        self.max_pool_w1 = nn.MaxPool1d(kernel_size=window_size - kernel_sizes[0] + 1, stride=0)
        self.max_pool_w2 = nn.MaxPool1d(kernel_size=window_size - kernel_sizes[1] + 1, stride=0)
        self.max_pool_w3 = nn.MaxPool1d(kernel_size=window_size - kernel_sizes[2] + 1, stride=0)

        # Linear layer
        self.linear_input_dim = (out_channels[0] + out_channels[1] + out_channels[2]) * 2
        self.linear1 = nn.Linear(self.linear_input_dim, self.linear_size)
        self.linear2 = nn.Linear(self.linear_size, 1)
    # end __init__

    # Forward for one side
    def forward_side(self, x):
        """
        Forward for one side
        :param x:
        :return:
        """
        # Embeddings
        embeds = self.embeddings(x)

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
        return out.view(-1, self.linear_input_dim / 2)
    # end forward_side

    # Forward
    def forward(self, x):
        """
        Forward
        :param x:
        :return:
        """
        # Each side
        x1 = x[:, self.window_size]
        x2 = x[:, :self.window_size]

        # CNN on one side
        out1 = self.forward_side(x1)
        out2 = self.forward_side(x2)

        # Concatenate
        out = torch.cat((out1, out2), dim=1)

        # Flatten
        out = out.view(-1, self.linear_input_dim)

        # Linear 1
        out = F.relu(self.linear1(out))

        # Linear 2
        out = F.relu(self.linear2(out))

        # Log Softmax
        return out
    # end forward

# end CNNCDist


# Load model
def cnncdist(pretrained=False, lang='en', **kwargs):
    """
    Load model
    :param pretrained:
    :param lang:
    :param kwargs:
    :return:
    """
    model = CNNCDist(**kwargs)
    voc = dict()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['cnncdist'][lang]))
        voc = model_zoo.load_url(voc_urls['cnncdist'][lang])
    # end if
    return model, voc
# end cnncdist
