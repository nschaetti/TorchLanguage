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
from random import shuffle, seed
import spacy
import re


# SFGram dataset
class SFGramDataset(Dataset):
    """
    SFGram dataset
    """

    # Constructor
    def __init__(self, author, root='./data', download=False, transform=None, load_type='wv'):
        """
        Constructor
        :param root: Data root directory.
        :param download: Download the dataset?
        :param dataset_size: How many samples from each author to load (1 to 100).
        :param transform: A TextTransformer object to apply.
        """
        # Properties
        self.root = root
        self.transform = transform
        self.author2id = dict()
        self.id2author = dict()
        self.texts = list()
        self.fold = 0
        self.authors_info = dict()
        self.n_authors = 0
        self.id2tag = dict()
        self.tag2id = dict()
        self.author = author
        self.load_type = load_type

        # Create directory if needed
        if not os.path.exists(self.root):
            self._create_root()
        # end if

        # Download the data set
        if download and not os.path.exists(os.path.join(self.root, "authors.json")):
            self._download()
        # end if

        # Generate data set
        self._load()
    # end __init__

    #############################################
    # PUBLIC
    #############################################

    # Segment text by authors
    def segment_text(self, text_content):
        """
        Segment text
        :param text_content:
        :return:
        """
        # List of segments
        segments = list()

        # The regex
        p = re.compile("SFGRAM\_(START|STOP)\_([A-Za-z]+)")

        # Last position
        last_position = 0

        # Author
        last_author = 'NONE'

        # For each find
        for i, m in enumerate(p.finditer(text_content)):
            segments.append((text_content[last_position:m.start()], last_author))
            # If start
            if u"START" in m.group():
                last_author = m.group()[m.group().rfind('_')+1:]
            elif u"STOP" in m.group():
                last_author = 'NONE'
            # end if

            # Last position
            last_position = m.start() + len(m.group())
        # end for

        # Add last segment
        segments.append((text_content[last_position:], last_author))

        return segments
    # end segment_text

    # Transform text
    def transform_text(self, text_path, n, text_content, author_prob):
        """
        Transform text
        :return:
        """
        # Transform
        if self.transform is not None:
            # Path to transformation
            path_transform = "{}.{}.{}.p".format(text_path, self.load_type, n)

            # Apply or load
            if os.path.exists(path_transform):
                transformed = torch.load(path_transform)
            else:
                transformed = self.transform(text_content)
                torch.save(transformed, path_transform)
            # end if

            # Transformed size
            if type(transformed) is list:
                transformed_size = len(transformed)
            elif type(transformed) is torch.LongTensor or type(transformed) is torch.FloatTensor \
                    or type(transformed) is torch.cuda.LongTensor or type(transformed) is torch.cuda.FloatTensor \
                    or type(transformed) is torch.Tensor:
                transformed_dim = transformed.dim()
                transformed_size = transformed.size(transformed_dim - 2)
            # end if

            return transformed, author_prob, self._create_labels(author_prob, transformed_size)
        else:
            return text_content, author_prob
        # end if
    # end transform_text

    #############################################
    # OVERRIDE
    #############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.texts)
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # Current file
        text_path = self.texts[idx]

        # Read text
        text_content = codecs.open(text_path, 'r', encoding='utf-8').read()

        # Segment text
        segments = self.segment_text(text_content)

        # Transform each segment
        for i, segment in enumerate(segments):
            # Segment
            segment_text, segment_author = segment

            # Ok
            if len(segment_text) > 0:
                # Transform
                if segment_author == self.author:
                    segment_transformed, _, segment_labels = self.transform_text(text_path, i, segment_text, 1.0)
                else:
                    segment_transformed, _, segment_labels = self.transform_text(text_path, i, segment_text, 0.0)
                # end if

                # Concate
                if i == 0:
                    text_transformed = segment_transformed
                    text_labels = segment_labels
                else:
                    text_transformed = torch.cat((text_transformed, segment_transformed), dim=0)
                    text_labels = torch.cat((text_labels, segment_labels), dim=0)
                # end if
            # end if
        # end for

        return text_transformed, text_labels
    # end __getitem__

    ##############################################
    # PRIVATE
    ##############################################

    # Create labels
    def _create_labels(self, author_prob, transformed_length):
        """
        Create labels
        :param author_name:
        :param length:
        :return:
        """
        # Vector
        tag_vector = torch.zeros(transformed_length, 1)

        # Set
        tag_vector[:, 0] = author_prob

        return tag_vector
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
        path_to_zip = os.path.join(self.root, "sfgram.zip")

        # Download
        urllib.urlretrieve("http://www.nilsschaetti.com/datasets/sfgram.zip", path_to_zip)

        # Unzip
        zip_ref = zipfile.ZipFile(path_to_zip, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()

        # Delete zip
        os.remove(path_to_zip)
    # end _download

    # Load dataset
    def _load(self):
        """
        Load the dataset
        :return:
        """
        # Authors info
        self.authors_info = json.load(open(os.path.join(self.root, "authors.json"), 'r'))
        self.n_authors = len(self.authors_info.keys())

        # For each author
        for author_index, author_name in enumerate(self.authors_info.keys()):
            self.author2id[author_name] = author_index
            self.id2author[author_index] = author_name
            self.id2tag[author_index] = self.authors_info[author_name]['id']
            self.tag2id[self.authors_info[author_name]['id']] = author_index
        # end for

        # For each text
        for f_index, file_name in enumerate(os.listdir(self.root)):
            if file_name[-4:] == ".txt":
                self.texts.append(os.path.join(self.root, file_name))
            # end if
        # end for

        # Shuffle texts
        seed(1)
        shuffle(self.texts)
    # end _load

# end ReutersC50Dataset
