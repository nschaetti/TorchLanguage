# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


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
    def __init__(self, window_size, vocab_size, n_classes, embedding_dim=50, out_channels=(500, 500, 500),
                 kernel_sizes=(3, 4, 5), n_linear=2, linear_size=1500, temporal_division=1.0, use_dropout=True):
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
        self.n_classes = n_classes
        self.n_linear = n_linear
        self.temporal_division = temporal_division
        self.use_dropout = use_dropout

        # Drop out
        self.dropout = nn.Dropout()
        self.dropout2d = nn.Dropout2d(p=0.7)

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

        # Max pooling sizes
        self.max_pool1_size = int(math.floor((window_size - kernel_sizes[0] + 1) / temporal_division))
        self.max_pool2_size = int(math.floor((window_size - kernel_sizes[1] + 1) / temporal_division))
        self.max_pool3_size = int(math.floor((window_size - kernel_sizes[2] + 1) / temporal_division))

        # Max pooling stride
        self.stride1 = self.max_pool1_size
        self.stride2 = self.max_pool2_size
        self.stride3 = self.max_pool3_size

        # Max pooling layer
        self.max_pool_w1 = nn.MaxPool1d(kernel_size=self.max_pool1_size, stride=self.stride1)
        self.max_pool_w2 = nn.MaxPool1d(kernel_size=self.max_pool2_size, stride=self.stride2)
        self.max_pool_w3 = nn.MaxPool1d(kernel_size=self.max_pool3_size, stride=self.stride3)

        # Final max pool size
        self.final_max_pool_size = (out_channels[0] + out_channels[1] + out_channels[2]) * temporal_division

        # Linear layer
        if self.n_linear == 2:
            self.linear_input_dim = self.final_max_pool_size * 2
            self.linear1 = nn.Linear(self.linear_input_dim, self.linear_size)
            self.linear2 = nn.Linear(self.linear_size, n_classes)
        elif self.n_linear == 1:
            self.linear_input_dim = (out_channels[0] + out_channels[1] + out_channels[2]) * 2
            self.linear1 = nn.Linear(self.linear_input_dim, n_classes)
        else:
            raise NotImplementedError(u"More than 2 layers not implemented")
        # end if
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
        """if self.use_dropout:
            out_win1 = self.dropout2d(self.conv_w1(embeds))
            out_win2 = self.dropout2d(self.conv_w2(embeds))
            out_win3 = self.dropout2d(self.conv_w3(embeds))
        else:
            out_win1 = self.conv_w1(embeds)
            out_win2 = self.conv_w2(embeds)
            out_win3 = self.conv_w3(embeds)
        # end if"""
        out_win1 = self.conv_w1(embeds)
        out_win2 = self.conv_w2(embeds)
        out_win3 = self.conv_w3(embeds)

        # Remove last dim
        out_win1 = torch.squeeze(out_win1, dim=3)
        out_win2 = torch.squeeze(out_win2, dim=3)
        out_win3 = torch.squeeze(out_win3, dim=3)

        # Max pooling
        max_win1 = F.relu(self.max_pool_w1(out_win1))
        max_win2 = F.relu(self.max_pool_w2(out_win2))
        max_win3 = F.relu(self.max_pool_w3(out_win3))

        # Concatenate
        out = torch.cat((max_win1, max_win2, max_win3), dim=1)

        # Flatten
        return out.view(-1, self.final_max_pool_size)
    # end forward_side

    # Forward
    def forward(self, x):
        """
        Forward
        :param x:
        :return:
        """
        # Each side
        x1 = x[:, self.window_size:]
        x2 = x[:, :self.window_size]

        # CNN on one side
        out1 = self.forward_side(x1)
        out2 = self.forward_side(x2)

        # Concatenate
        out = torch.cat((out1, out2), dim=1)

        # Flatten
        out = out.view(-1, self.linear_input_dim)

        # Linear
        if self.n_linear == 2:
            # Linear 1
            out = F.relu(self.linear1(self.dropout(out)))

            # Linear 2
            return F.log_softmax(self.linear2(self.dropout(out)))
            # return self.linear2(self.dropout(out))
        else:
            return self.linear1(self.dropout(out))
        # end if
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
