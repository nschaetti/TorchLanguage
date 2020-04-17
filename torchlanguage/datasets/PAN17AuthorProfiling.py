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
    def __init__(self, lang, outputs_length, output_dim, load_type, output_type='float', n_tweets=100, root='./data',
                 trained=False, download=False, transform=None, shuffle=True, per_tweet=False, save_transform=False):
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
        self.load_type = load_type
        self.last_tokens = None
        self.tokenizer = torchlanguage.transforms.Token()
        self.lang = lang
        self.trained = trained
        self.user_tweets = list()
        self.ground_truths = dict()
        self.long_tweet = 0
        self.n_tweets = n_tweets
        self.output_type = output_type
        self.shuffle = shuffle
        self.per_tweet = per_tweet
        self.user2tweets = dict()
        self.save_transform = save_transform

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

    #region PUBLIC

    # Get item per tweet
    def get_item_per_tweet(self, idx):
        """
        Get item per tweet
        :param idx:
        :return:
        """
        # User and tweet position
        user_position = int(idx / 100.0)
        tweet_position = idx - (user_position * 100)

        # Current user
        user_id, _, _ = self.user_tweets[user_position]

        # Path to precomputed version
        precomputed_path = os.path.join(self.root, user_id + "." + self.load_type + ".xml")

        # Save tweets
        tweets_inputs, tweets_gender_outputs, tweets_country_outputs, tweets_lengths = torch.load(precomputed_path)

        # Current user
        user_id, user_gender, user_country = self.user_tweets[user_position]
        user_gender, user_country = self.ground_truths[user_id]

        # Total tweets
        total_tweets = self.user2tweets[user_id]

        # Last text
        self.last_tokens = self.tokenizer(total_tweets[-1])

        # Transform
        if self.transform is not None:
            return (tweets_inputs, user_gender, user_country, (tweets_gender_outputs, tweets_country_outputs), tweets_lengths)
        else:
            return (total_tweets, user_gender, user_country, tweets_lengths)
        # end if
    # end get_item_per_tweet

    # Get item per group
    def get_item_per_group(self, idx):
        """
        Get item per group
        :param idx:
        :return:
        """
        # Current user
        user_id, _, _ = self.user_tweets[idx]

        # Empty vector
        if not self.trained:
            output_vector = torch.zeros(self.n_tweets, self.outputs_length, self.output_dim)
        else:
            output_vector = torch.LongTensor(self.n_tweets, self.outputs_length).fill_(0)
        # end if

        # Length vector
        lengths_vector = torch.LongTensor(self.n_tweets)

        # Total tweets
        total_tweets = self.user2tweets[user_id]

        # Tweet length
        self.long_tweet = 0

        # For each tweet
        for tweet in total_tweets:
            if len(tweet) > self.long_tweet:
                self.long_tweet = len(tweet)
            # end if
        # end for

        # Last text
        self.last_tokens = self.tokenizer(total_tweets[-1])

        # Path to precomputed version
        precomputed_path = os.path.join(self.root, user_id + "." + self.load_type + ".xml")

        # Transform
        if self.transform is not None:
            # Precomputer exists
            if not os.path.exists(precomputed_path) or not self.save_transform:
                # Transform each tweets
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

                    # Final size
                    if transformed_size > self.outputs_length:
                        final_size = self.outputs_length
                    else:
                        final_size = transformed_size
                    # end if

                    # Add to outputs
                    output_vector[tweet_i, :final_size] = transformed[:final_size]

                    # Length of tweet
                    # lengths_vector[tweet_i] = len(tweet)
                    lengths_vector[tweet_i] = final_size
                # end for

                # Save if needed
                if self.save_transform:
                    # Save tweets
                    torch.save(
                        (output_vector, lengths_vector),
                        precomputed_path
                    )
                # end if
            else:
                # Load precomputed
                output_vector, lengths_vector = torch.load(precomputed_path)
            # end if

            # Return
            return (
                output_vector,
                self.ground_truths[user_id][0],
                self.ground_truths[user_id][1],
                self._create_labels(
                    self.ground_truths[user_id][0],
                    self.ground_truths[user_id][1]
                ),
                lengths_vector
            )
        else:
            return (
                total_tweets,
                self.ground_truths[user_id][0],
                self.ground_truths[user_id][1],
                lengths_vector
            )
        # end if
    # end get_item_per_group

    #endregion PUBLIC

    # region PRIVATE

    # Transform XML file to transformed inputs and outputs
    def _transform_xml(self, xml_path, gender, country):
        """
        Get item per group
        :param idx:
        :return:
        """
        # Read XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # User tweets
        total_tweets = list()

        # For each tweet
        for elem in root[0]:
            total_tweets.append(elem.text)
        # end for

        # Empty vector
        if not self.trained:
            output_vector = torch.zeros(self.n_tweets, self.outputs_length, self.output_dim)
        else:
            output_vector = torch.LongTensor(self.n_tweets, self.outputs_length).fill_(0)
        # end if

        # Length vector
        lengths_vector = torch.LongTensor(self.n_tweets)

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

                # Length of tweet
                lengths_vector[tweet_i] = len(tweet)
            # end for

            # Return
            return output_vector, self._create_labels(gender, country[:-1]), lengths_vector
        else:
            return (total_tweets, lengths_vector)
        # end if
    # end get_item_per_group

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

            # Path to precomputed version
            precomputed_path = os.path.join(self.root, user_id + "." + self.load_type + ".xml")
            xml_path = os.path.join(self.root, user_id + ".xml")

            # Read XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # User tweets
            total_tweets = list()

            # For each tweet
            for elem in root[0]:
                total_tweets.append(elem.text)
            # end for

            # Precomputed if necessary
            """if not os.path.exists(precomputed_path) and self.save_transform:
                # Transform tweets
                tweets_inputs, [tweets_gender_outputs, tweets_country_outputs], tweets_lengths = self._transform_xml(
                    xml_path,
                    user_gender,
                    user_country
                )

                # Save tweets
                torch.save(
                    (tweets_inputs, tweets_gender_outputs, tweets_country_outputs, tweets_lengths),
                    precomputed_path
                )
            # end if"""

            # Add
            self.user_tweets.append((user_id, user_gender, user_country[:-1]))
            self.ground_truths[user_id] = (user_gender, user_country[:-1])
            self.user2tweets[user_id] = total_tweets
        # end for

        # Shuffle list of users
        if self.shuffle:
            random.shuffle(self.user_tweets)
        # end if
    # end _load

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
            if self.per_tweet:
                # Gender vector
                gender_vector = torch.zeros(self.outputs_length, 2)
                gender_vector[:, gender_num] = 1.0

                # Country vector
                country_vector = torch.zeros(self.outputs_length, self.country_size[self.lang])
                country_vector[:, country_num] = 1.0
            else:
                # Gender vector
                gender_vector = torch.zeros(self.n_tweets, self.outputs_length, 2)
                gender_vector[:, :, gender_num] = 1.0

                # Country vector
                country_vector = torch.zeros(self.n_tweets, self.outputs_length, self.country_size[self.lang])
                country_vector[:, :, country_num] = 1.0
            # end if
        else:
            if self.per_tweet:
                # Gender vector
                gender_vector = torch.LongTensor(self.outputs_length).fill_(gender_num)

                # Country vector
                country_vector = torch.LongTensor(self.outputs_length).fill_(country_num)
            else:
                # Gender vector
                gender_vector = torch.LongTensor(self.n_tweets, self.outputs_length).fill_(gender_num)

                # Country vector
                country_vector = torch.LongTensor(self.n_tweets, self.outputs_length).fill_(country_num)
            # end if
        # end if

        return (gender_vector, country_vector)
    # end _create_labels

    # endregion PRIVATE

    #region OVERRIDE

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        if self.per_tweet:
            return len(self.user_tweets) * 100
        else:
            return len(self.user_tweets)
        # end if
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        if self.per_tweet:
            return self.get_item_per_tweet(idx)
        else:
            return self.get_item_per_group(idx)
        # end if
    # end __getitem__

    #endregion OVERRIDE

# end ReutersC50Dataset
