# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['CCSAA', 'ccsaa']


# Model URLs
model_urls = {
    'cgfs': {
        'en': 'https://www.nilsschaetti.com/models/ccsaa-en-bd63e232.pth',
        'fr': 'https://www.nilsschaetti.com/models/ccsaa-fr-4df8aa71.pth'
    }
}


# CNN Character Selector for Authorship Attribution
class CCSAA(nn.Module):
    """
    CNN Character Selector for Authorship Attribution
    """

    # Constructor
    def __init__(self, text_length, vocab_size, n_classes, embedding_dim=50, out_channels=(200, 200, 200),
                 kernel_sizes=(3, 4, 5), n_features=100, use_dropout=True):
        """
        Constructor
        :param vocab_size: Vocabulary size
        :param embedding_dim: Embedding layer dimension
        :param out_channels: Number of output channels
        :param kernel_sizes: Different kernel sizes
        """
        super(CCSAA, self).__init__()

        # Properties
        self.embedding_dim = embedding_dim
        self.text_length = text_length
        self.n_classes = n_classes
        self.use_dropout = use_dropout
        self.n_features = n_features

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
        self.max_pool_w1 = nn.MaxPool1d(kernel_size=text_length - kernel_sizes[0] + 1, stride=0)
        self.max_pool_w2 = nn.MaxPool1d(kernel_size=text_length - kernel_sizes[1] + 1, stride=0)
        self.max_pool_w3 = nn.MaxPool1d(kernel_size=text_length - kernel_sizes[2] + 1, stride=0)

        # Dropout
        self.drop_out = nn.Dropout()

        # Linear layer
        self.linear_size = out_channels[0] + out_channels[1] + out_channels[2]
        self.linear = nn.Linear(self.linear_size, self.n_features)

        # Output linear
        self.linear2 = nn.Linear(self.n_features, n_classes)
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

        # Dropout
        if self.use_dropout:
            out = self.drop_out(out)
        # end if

        # Linear
        out = F.relu(self.linear(out))

        # Normalize
        out = F.normalize(out, p=2, dim=1)

        # Output linear
        out = self.linear2(out)

        # Log Softmax
        # return F.log_softmax(out, dim=1)
        return out
    # end forward

# end CGFS


# Load model
def ccsaa(pretrained=False, lang='en', **kwargs):
    """
    Load model
    :param pretrained:
    :param lang:
    :param kwargs:
    :return:
    """
    model = CCSAA(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['ccsaa'][lang]))
    # end if
    return model
# end cgfs
