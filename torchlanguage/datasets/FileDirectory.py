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
import os
import codecs


# File directory dataset
class FileDirectory(Dataset):
    """
    Load files from a directory
    """

    # Constructor
    def __init__(self, root='./data', download=False, transform=None, download_url=""):
        """
        Constructor
        :param root: Data root directory.
        :param download: Download the dataset?
        :param transform: A TextTransformer object to apply.
        """
        # Properties
        self.root = root
        self.transform = transform

        # Create directory if needed
        if not os.path.exists(self.root):
            self._create_root()
        # end if

        # Download the data set
        if download and download_url != "":
            self._download()
        # end if

        # List file
        self.files = os.listdir(root)
    # end __init__

    #############################################
    # PUBLIC
    #############################################

    #############################################
    # PRIVATE
    #############################################

    # Create the root directory
    def _create_root(self):
        """
        Create the root directory
        :return:
        """
        os.mkdir(self.root)
    # end _create_root

    # Download the dataset
    def _download(self):
        """
        Downlaod the dataset
        :return:
        """
        # Path to zip file
        path_to_zip = os.path.join(self.root, "dataset.zip")

        # Download
        urllib.urlretrieve("http://www.nilsschaetti.com/datasets/dataset.zip", path_to_zip)

        # Unzip
        zip_ref = zipfile.ZipFile(path_to_zip, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()

        # Delete zip
        os.remove(path_to_zip)
    # end _download

    #############################################
    # OVERRIDE
    #############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.files)
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        return self.transform(codecs.open(self.files[idx], 'rb', encoding='utf-8').read())
    # end __getitem__

# end FileDirectory
