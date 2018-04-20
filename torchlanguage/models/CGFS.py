# -*- coding: utf-8 -*-
#

# Imports
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['CGFS', 'cgfs']


# Model URLs
model_urls = {
    'cgfs': {
        'en': 'https://www.nilsschaetti.com/models/cgfs-en-bd63e232.pth',
        'fr': 'https://www.nilsschaetti.com/models/cgfs-fr-4df8aa71.pth'
    }
}


# CNN Glove Feature Selector
class CGFS(nn.Module):
    """
    CNN Glove Feature Selector
    """

    # Constructor
    def __init__(self, n_gram, n_authors=15, out_channels=(30, 20), kernel_sizes=(5, 5), max_pool_size=2, n_features=30):
        """
        Constructor
        :param n_gram:
        :param n_authors:
        :param out_channels:
        :param kernel_sizes:
        :param max_pool_size:
        :param n_features:
        """
        super(CGFS, self).__init__()
        self.n_features = n_features
        self.n_authors = n_authors

        # Conv 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels[0], kernel_size=(n_gram, kernel_sizes[0]))

        # Max pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=(1, max_pool_size), stride=0)

        # Conv 2
        self.conv2 = nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_sizes[1])

        # Linear layer 1
        self.linear_size = out_channels[1] * 72
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
        # Conv 1
        out_conv1 = F.relu(self.conv1(x))

        # Max pooling
        max_pooled = self.max_pool(out_conv1)

        # Remove dim
        max_pooled = max_pooled.squeeze(2)

        # Conv 2
        out_conv2 = F.relu(self.conv2(max_pooled))

        # Max pooling
        max_pooled = self.max_pool(out_conv2)

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

# end CGFS


# Load model
def cgfs(pretrained=False, lang='en', **kwargs):
    """
    Load model
    :param pretrained:
    :param lang:
    :param kwargs:
    :return:
    """
    model = CGFS(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['cgfs'][lang]))
    # end if
    return model
# end cgfs
