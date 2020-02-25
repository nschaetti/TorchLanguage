# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import torchlanguage.transforms
import urllib.request
import os
import zipfile
import xml.etree.ElementTree as ET
import random


# PAN 17 Author Profiling
class PAN17AuthorProfiling(Dataset):
    """
    PAN 17 Author Profiling
    """

    # Constructor
    def __init__(self, lang, outputs_length, output_dim, output_type='float', n_tweets=100, root='./data',
                 download=False, transform=None, shuffle=False):
        """
        Constructor
        :param lang: Which subset to load.
        :param outputs_length: Output vector length.
        :param output_dim: Output dimension
        :param root: Data root directory.
        :param download: Download the dataset?
        :param transform: A TextTransformer object to apply.
        """
        # Properties
        self.root = root
        self.outputs_length = outputs_length
        self.output_dim = output_dim
        self.transform = transform
        self.last_tokens = None
        self.tokenizer = torchlanguage.transforms.Token()
        self.lang = lang
        self.user_tweets = list()
        self.ground_truths = dict()
        self.long_tweet = 0
        self.n_tweets = n_tweets
        self.output_type = output_type
        self.shuffle = True

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
        user_id, _, _ = self.user_tweets[idx]

        # Empty vector
        output_vector = torch.zeros(self.n_tweets, self.outputs_length, self.output_dim)

        # Read XML file
        tree = ET.parse(os.path.join(self.root, user_id + ".xml"))
        root = tree.getroot()

        # Total tweets
        total_tweets = list()

        # Longest tweet
        self.long_tweet = 0

        # For each tweet
        for elem in root[0]:
            total_tweets.append(elem.text)
            if len(elem.text) > self.long_tweet:
                self.long_tweet = len(elem.text)
            # end if
        # end for

        # Last text
        self.last_tokens = self.tokenizer(total_tweets[-1])

        # Transform
        if self.transform is not None:
            # For each tweets
            for tweet_i, tweet in enumerate(total_tweets):
                # Transform
                transformed = self.transform(tweet)

                # Transformed size
                if type(transformed) is list:
                    transformed_size = len(transformed)
                elif type(transformed) is torch.LongTensor or type(transformed) is torch.FloatTensor \
                        or type(transformed) is torch.cuda.LongTensor or type(transformed) is torch.cuda.FloatTensor \
                        or type(transformed) is torch.Tensor:
                    transformed_dim = transformed.dim()
                    transformed_size = transformed.size(transformed_dim - 2)
                else:
                    raise Exception("Unknown transformed type")
                # end if

                # Add to outputs
                output_vector[tweet_i, :transformed_size] = transformed
            # end for

            return (
                output_vector,
                self.ground_truths[user_id][0],
                self.ground_truths[user_id][1],
                self._create_labels(
                    self.ground_truths[user_id][0],
                    self.ground_truths[user_id][1]
                )
            )
        else:
            return (
                total_tweets,
                self.ground_truths[user_id][0],
                self.ground_truths[user_id][1]
            )
        # end if
    # end __getitem__

    #region PRIVATE

    # Create labels
    def _create_labels(self, user_gender, user_country):
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

        if self.output_type == 'float':
            # Gender vector
            gender_vector = torch.zeros(self.n_tweets, self.outputs_length, 2)
            gender_vector[:, :, gender_num] = 1.0

            # Country vector
            country_vector = torch.zeros(self.n_tweets, self.outputs_length, self.country_size[self.lang])
            country_vector[:, :, country_num] = 1.0
        else:
            # Gender vector
            gender_vector = torch.LongTensor(self.n_tweets, self.outputs_length).fill_(gender_num)

            # Country vector
            country_vector = torch.LongTensor(self.n_tweets, self.outputs_length).fill_(country_num)
        # end if

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
        urllib.request.urlretrieve("http://www.nilsschaetti.com/datasets/pan17-author-profiling.zip", path_to_zip)

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
            self.user_tweets.append((user_id, user_gender, user_country[:-1]))
            self.ground_truths[user_id] = (user_gender, user_country[:-1])
        # end for

        # Shuffle list of users
        if self.shuffle:
            random.shuffle(self.user_tweets)
        # end if
    # end _load

    #endregion PRIVATE

# end ReutersC50Dataset
