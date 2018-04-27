# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import autograd


__all__ = ['BiEGRU', 'biegru']


# Model URLs
model_urls = {
    'style-change-detection': {
        'en': 'https://www.nilsschaetti.com/models/biegru-en-bd63e232.pth'
    }
}


# Bidirectional Embedded GRU (Text)
class BiEGRU(nn.Module):
    """
    Bidirectional Embedded GRU
    """

    # Constructor
    def __init__(self, window_size, vocab_size, hidden_dim, n_classes, embedding_dim=50, out_channels=(20, 20, 20),
                 kernel_sizes=(3, 4, 5)):
        """
        Constructor
        :param vocab_size: Vocabulary size
        :param embedding_dim: Embedding layer dimension
        :param out_channels: Number of output channels
        :param kernel_sizes: Different kernel sizes
        """
        super(BiEGRU, self).__init__()

        # Properties
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.hidden_size = hidden_dim

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

        # GRU input size
        self.gru_input_size = out_channels[0] + out_channels[1] + out_channels[2]

        # Bidirectional GRU layer
        self.bi_grus = torch.nn.GRU(input_size=self.gru_input_size, hidden_size=hidden_dim, num_layers=1,
                                    batch_first=False, bidirectional=True)

        # GRU layer
        self.reverse_gru = torch.nn.GRU(input_size=self.gru_input_size, hidden_size=hidden_dim, num_layers=1,
                                        batch_first=False, bidirectional=False)

        # Linear dense layer
        self.linear_layer = torch.nn.Linear(in_features=hidden_dim * 2, out_features=n_classes)
    # end __init__

    # Init hidden layer
    def init_hidden(self, batch_size):
        """
        Init hidden layer
        :return:
        """
        return autograd.Variable(torch.randn(2, batch_size, self.hidden_size))
    # end init_hidden

    # Reset hidden
    def reset_hidden(self, h):
        """
        Reset hidden
        :return:
        """
        return h.fill_(0.0)
    # end reset_hidden

    # Forward
    def forward(self, x, h):
        """
        Forward
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
        out = out.view(-1, 1, self.gru_input_size)

        # GRU
        gru_output, hidden = self.bi_grus(out, h)

        # Linear
        out = self.linear_layer(gru_output[-1])

        # Log Softmax
        return F.log_softmax(out, dim=1), hidden
    # end forward

# end BiEGRU


# Load model
def biegru(pretrained=False, lang='en', task='style-change-detection', **kwargs):
    """
    Load model
    :param pretrained:
    :param lang:
    :param kwargs:
    :return:
    """
    model = BiEGRU(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[task][lang]))
    # end if
    return model
# end biegru

