# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import urllib.request
import os
import zipfile
import json
import codecs
from random import shuffle, seed
import re
import math


# SFGram block dataset (pretrained features)
class SFGramBlockDataset(Dataset):
    """
    SFGram dataset (pretrained features)
    """

    # Constructor
    def __init__(self, author, root='./sfgram_block', feature='wv', trained=False, block_length=40, n_files=91):
        """
        Constructor
        :param author: Author to load
        :param root: Where are the data files
        :param feature: Feature to load (wv, c1, c2, c3)
        :param trained: Embeddings are trained
        :param block_length: Block length to return
        """
        # Properties
        self.root = root
        self.author_root = os.path.join(root, author)
        self.data_inputs = None
        self.data_outputs = None
        self.data_length = 0
        self.dataset_size = 0
        self.feature = feature
        self.trained = trained
        self.block_length = block_length
        self.load_type = self.feature + "T" if self.trained else ""
        self.trues_count = 0
        self.input_dim = 0
        self.data_dim = 0
        self.data_path = os.path.join(self.author_root, self.load_type)
        self.n_files = n_files

        # Generate data set
        self._load()
    # end __init__

    #region PUBLIC

    #endregion PUBLIC

    # region PRIVATE

    # Load dataset
    def _load(self):
        """
        Load the dataset
        :return:
        """
        # Load type
        # Load JSON info file
        data_info_path = os.path.join(self.author_root, "sfgam_info.{}.json".format(self.load_type))
        data_info = json.load(open(data_info_path, "r"))

        # Load info
        self.data_length = data_info['8033959']
        self.dataset_size = int(self.data_length / self.block_length)
        self.trues_count = int(data_info['trues'])
        self.data_dim = data_info['tensor_dim']
        self.input_dim = data_info['input_dim']

        # Create input data tensor
        if data_info['tensor_type'] == 'float':
            self.data_inputs = torch.zeros(self.data_length, self.input_dim)
        else:
            self.data_inputs = torch.LongTensor(self.data_length).fill_(0)
        # end if

        # Create output data tensor
        self.data_outputs = torch.zeros(self.data_length)

        # Position in global tensor
        t = 0

        # For each file to load
        for data_file_i in range(self.n_files):
            # Path to the data file
            data_file_path = os.path.join(self.data_path, "sfgram.{}.p".format(data_file_i))

            # Load file with torch
            (document_inputs, document_outputs) = torch.load(data_file_path)

            # Save
            self.data_inputs[t] = document_inputs
            self.data_outputs[t] = document_outputs

            # Position
            t += document_inputs.size(0)
        # end for
    # end _load

    # endregion PRIVATE

    #region OVERRIDE

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return self.dataset_size
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # Position
        pos_start = idx * self.block_length
        pos_end = pos_start + self.block_length

        # Return
        return self.data_inputs[pos_start:pos_end], self.data_outputs[pos_start:pos_end]
    # end __getitem__

    #endregion OVERRIDE

# end SFGramDataset
