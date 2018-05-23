# -*- coding: utf-8 -*-
#

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['CNNCTweet', 'cnnctweet']


# Model settings
model_settings = {
    'c1': {
    },
    'c2': {
        'en': {
            'model': 'https://www.nilsschaetti.com/models/cnnctweet-en-0abe5371.pth',
            'voc': 'https://www.nilsschaetti.com/models/cnnctweet-voc-en-83ebc7f3.pth',
            'embedding_dim': 50,
            'voc_size': 21510,
            'text_length': 165
        },
        'es': {
            'model': 'https://www.nilsschaetti.com/models/cnnctweet-es-1a993977.pth',
            'voc': 'https://www.nilsschaetti.com/models/cnnctweet-voc-es-43d0a2a7.pth',
            'embedding_dim': 50,
            'voc_size': 30025,
            'text_length': 165
        },
        'ar': {
            'model': 'https://www.nilsschaetti.com/models/cnnctweet-ar-9f761a33.pth',
            'voc': 'https://www.nilsschaetti.com/models/cnnctweet-voc-ar-5fb3b776.pth',
            'embedding_dim': 50,
            'voc_size': 31694,
            'text_length': 165
        }
    }
}


# CNNC-Tweet (Text)
class CNNCTweet(nn.Module):
    """
    CNNC on tweet
    """

    # Constructor
    def __init__(self, text_length, vocab_size, embedding_dim=300, out_channels=(500, 500, 500),
                 kernel_sizes=(3, 4, 5), n_classes=2):
        """
        Constructor
        :param vocab_size: Vocabulary size
        :param embedding_dim: Embedding layer dimension
        :param out_channels: Number of output channels
        :param kernel_sizes: Different kernel sizes
        """
        super(CNNCTweet, self).__init__()

        # Properties
        self.embedding_dim = embedding_dim
        self.text_length = text_length
        self.n_classes = n_classes

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

        # Linear layer
        self.linear_size = out_channels[0] + out_channels[1] + out_channels[2]
        self.linear = nn.Linear(self.linear_size, n_classes)
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

        # Linear
        out = self.linear(out)

        # Log Softmax
        return F.log_softmax(out, dim=1)
    # end forward

# end CNNCTweet


# Load model
def cnnctweet(pretrained=False, n_gram='c2', lang='en', map_location=None, **kwargs):
    """
    Load model
    :param pretrained:
    :param lang:
    :param kwargs:
    :return:
    """
    if pretrained:
        model = CNNCTweet(text_length=model_settings[n_gram][lang]['text_length'], vocab_size=model_settings[n_gram][lang]['voc_size'],
                          embedding_dim=model_settings[n_gram][lang]['embedding_dim'])
        model.load_state_dict(model_zoo.load_url(model_settings[n_gram][lang]['model'], map_location=map_location))
        voc = model_zoo.load_url(model_settings[n_gram][lang]['voc'], map_location=map_location)
    else:
        model = CNNCTweet(**kwargs)
        voc = dict()
    # end if
    return model, voc
# end cnnctweet
