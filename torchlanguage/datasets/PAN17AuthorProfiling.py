# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import torchlanguage.transforms
import urllib
import os
import zipfile
import json
import codecs
import random
import pickle
from datetime import datetime
import xml.etree.ElementTree as ET


# PAN 17 Author Profiling
class PAN17AuthorProfiling(Dataset):
    """
    PAN 17 Author Profiling
    """

    # Constructor
    def __init__(self, lang, root='./data', download=False, transform=None):
        """
        Constructor
        :param lang: Which subset to load.
        :param root: Data root directory.
        :param download: Download the dataset?
        :param transform: A TextTransformer object to apply.
        """
        # Properties
        self.root = root
        self.transform = transform
        self.last_tokens = None
        self.tokenizer = torchlanguage.transforms.Token()
        self.lang = lang
        self.user_tweets = list()
        self.ground_truths = dict()

        # To num
        self.gender2num = {'male': 0, 'female': 1}
        self.country2num = {'en': {'canada': 0, 'australia': 1, 'new zealand': 2, 'ireland': 3, 'great britain': 4, 'united states': 5},
                            'ar': {'gulf': 0, 'levantine': 1, 'maghrebi': 2, 'egypt': 3},
                            'pt': {'portugal': 0, 'brazil': 1},
                            'es': {'colombia': 0, 'argentina': 1, 'spain': 2, 'venezuela': 3, 'peru': 4, 'chile': 5, 'mexico': 6}}
        self.country_size = {'en': 6, 'ar': 4, 'pt': 2, 'es': 7}

        # Create directory if needed
        if not os.path.exists(self.root):
            self._create_root()
        # end if

        # Download the data set
        if download and not os.path.exists(os.path.join(self.root, lang + ".txt")):
            self._download()
        # end if

        # Generate data set
        self._load()
    # end __init__

    #############################################
    # OVERRIDE
    #############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.user_tweets)
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # Current user
        user_id = self.user_tweets[idx]

        # Read XML file
        tree = ET.parse(os.path.join(self.root, user_id + ".xml"))
        root = tree.getroot()

        # Total tweets
        total_tweets = u""

        # For each tweet
        for elem in root:
            total_tweets += elem.text + u" "
        # end for

        # Last text
        self.last_tokens = self.tokenizer(total_tweets)

        # Transform
        if self.transform is not None:
            # Transform
            transformed = self.transform(total_tweets)

            # Transformed size
            if type(transformed) is list:
                transformed_size = len(transformed)
            elif type(transformed) is torch.LongTensor or type(transformed) is torch.FloatTensor \
                    or type(transformed) is torch.cuda.LongTensor or type(transformed) is torch.cuda.FloatTensor \
                    or type(transformed) is torch.Tensor:
                transformed_dim = transformed.dim()
                transformed_size = transformed.size(transformed_dim - 2)
            # end if

            return (transformed,
                    self.ground_truths[user_id][0],
                    self.ground_truths[user_id][1],
                    self._create_labels(self.ground_truths[user_id][0], self.ground_truths[user_id][1], transformed_size))
        else:
            return (total_tweets,
                    self.ground_truths[user_id][0],
                    self.ground_truths[user_id][1])
        # end if
    # end __getitem__

    ##############################################
    # PRIVATE
    ##############################################

    # Create labels
    def _create_labels(self, user_gender, user_country, transformed_length):
        """
        Create labels
        :param user_gender:
        :param user_country:
        :param transformed_length:
        :return:
        """
        # Label to id
        gender_num = self.gender2num[user_gender]
        country_num = self.country2num[self.lang][user_country]

        # Gender vector
        gender_vector = torch.zeros(transformed_length, 2)
        gender_vector[:, gender_num] = 1.0

        # Country vector
        country_vector = torch.zeros(transformed_length, self.country_size[self.lan])
        country_vector[:, country_num] = 1.0

        return (gender_vector, country_vector)
    # end _create_labels

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
        path_to_zip = os.path.join(self.root, "pan17-author-profiling.zip")

        # Download
        urllib.urlretrieve("http://www.nilsschaetti.com/datasets/pan17-author-profiling.zip", path_to_zip)

        # Unzip
        zip_ref = zipfile.ZipFile(path_to_zip, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()

        # Delete zip
        os.remove(path_to_zip)
    # end _download

    # Load information
    def _load(self):
        """
        Load the dataset
        :return:
        """
        # Dataset info
        lines = open(os.path.join(self.root, self.lang + ".txt"), 'r').readlines()

        # For each line
        for line in lines:
            # Split
            user_id, user_gender, user_country = line.split(":::")

            # Add
            self.user_tweets.append((user_id, user_gender, user_country))
            self.ground_truths[user_id] = (user_gender, user_country)
        # end for
    # end _load

# end ReutersC50Dataset
