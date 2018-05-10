# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


__all__ = ['CNNSCD', 'cnnscd', 'cnnscd25']


# Model settings
model_settings = {
    'c1': {
        25: {
            'en': {
                'model': 'https://www.nilsschaetti.com/models/cnnscd-25-en-34b74a3f.pth',
                'voc': 'https://www.nilsschaetti.com/models/cnnscd-25-voc-en-3875f0ad.pth',
                'embedding_dim': 50,
                'voc_size': 1628,
                'text_length': 12000
            }
        }
    },
    'c2': {
    }
}


# CNN on character for distance learning
class CNNSCD(nn.Module):
    """
    CNN on character for distance learning
    """

    # Constructor
    def __init__(self, input_dim, vocab_size, embedding_dim=50, out_channels=(25, 25, 25), kernel_sizes=(3, 4, 5),
                 n_linear=1, linear_size=1500, max_pool_size=700, max_pool_stride=350, use_dropout=True):
        """
        Constructor
        :param input_dim:
        :param vocab_size:
        :param embedding_dim:
        :param out_channels:
        :param kernel_sizes:
        :param n_linear:
        :param linear_size:
        :param max_pool_size:
        :param max_pool_stride:
        """
        super(CNNSCD, self).__init__()

        # Properties
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.linear_size = linear_size
        self.n_linear = n_linear
        self.max_pool_size = max_pool_size
        self.max_pool_stride = max_pool_stride
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

        # Max pooling layer
        self.max_pool_w1 = nn.MaxPool1d(kernel_size=self.max_pool_size, stride=self.max_pool_stride)
        self.max_pool_w2 = nn.MaxPool1d(kernel_size=self.max_pool_size, stride=self.max_pool_stride)
        self.max_pool_w3 = nn.MaxPool1d(kernel_size=self.max_pool_size, stride=self.max_pool_stride)

        # Max pooling layer outputs
        self.max_pool1_output_dim = (
                    out_channels[0] * int(math.ceil((input_dim - self.max_pool_size) / float(self.max_pool_stride))))
        self.max_pool2_output_dim = (
                    out_channels[1] * int(math.ceil((input_dim - self.max_pool_size) / float(self.max_pool_stride))))
        self.max_pool3_output_dim = (
                out_channels[2] * int(math.ceil((input_dim - self.max_pool_size) / float(self.max_pool_stride))))

        # Max pooling output dim
        self.max_pool_output_dim = self.max_pool1_output_dim + self.max_pool2_output_dim + self.max_pool3_output_dim

        # Linear layer
        if self.n_linear == 2:
            self.linear1 = nn.Linear(self.max_pool_output_dim, self.linear_size)
            self.linear2 = nn.Linear(self.linear_size, 2)
        elif self.n_linear == 1:
            self.linear1 = nn.Linear(self.max_pool_output_dim, 2)
        else:
            raise NotImplementedError(u"More than 2 layers not implemented")
        # end if
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

        # Flatten
        max_win1 = max_win1.view(-1, self.max_pool1_output_dim)
        max_win2 = max_win2.view(-1, self.max_pool2_output_dim)
        max_win3 = max_win3.view(-1, self.max_pool3_output_dim)

        # Concatenate
        out = torch.cat((max_win1, max_win2, max_win3), dim=1)

        # Linear
        if self.n_linear == 2:
            # Linear 1
            out = F.relu(self.linear1(self.dropout(out)))

            # Linear 2
            return self.linear2(self.dropout(out))
        else:
            return self.linear1(self.dropout(out))
        # end if
    # end forward

# end CNNSCD


# Load model
def cnnscd25(n_gram='c1', lang='en', map_location=None):
    """
    Load model
    :param n_gram:
    :param lang:
    :param map_location:
    :return:
    """
    model = CNNSCD(
        input_dim=model_settings[n_gram][25][lang]['text_length'],
        vocab_size=model_settings[n_gram][25][lang]['voc_size'],
        out_channels=(25, 25, 25),
        embedding_dim=model_settings[n_gram][25][lang]['embedding_dim'],
        n_linear=2,
        linear_size=1500
    )
    model.load_state_dict(model_zoo.load_url(model_settings[n_gram][25][lang]['model'], map_location=map_location))
    voc = model_zoo.load_url(model_settings[n_gram][25][lang]['voc'], map_location=map_location)
    return model, voc
# end cnnscd25


# Load model
def cnnscd(**kwargs):
    """
    Load model
    :param kwargs:
    :return:
    """
    model = CNNSCD(**kwargs)
    voc = dict()
    return model, voc
# end cnnscd
