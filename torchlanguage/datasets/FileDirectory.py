# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import urllib
import os
import zipfile
import json
import codecs
from random import shuffle
import math
import pickle
from datetime import datetime


# File directory dataset
class FileDirectory(Dataset):
    """
    Load files from a directory
    """

    # Constructor
    def __init__(self, root='./data', download="", transform=None):
        """
        Constructor
        :param root: Data root directory.
        :param download: Download the dataset?
        :param transform: A TextTransformer object to apply.
        """
        pass
    # end __init__

    #############################################
    # PUBLIC
    #############################################

    #############################################
    # OVERRIDE
    #############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        pass
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        pass
    # end __getitem__

# end ReutersC50Dataset
