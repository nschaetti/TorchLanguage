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
import spacy


# SFGram dataset
class SFGramDataset(Dataset):
    """
    SFGram dataset
    """

    # Constructor
    def __init__(self, root='./data', download=False, transform=None, dataset_size=91, model='en_vectors_web_lg'):
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
        self.dataset_size = dataset_size
        self.n_authors = 0
        self.id2tag = dict()
        self.tag2id = dict()
        self.nlp = spacy.load(model)

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

    # Tag text
    def tag_text(self, text_content):
        """
        Tag text
        :param text_content:
        :return:
        """
        # Labels
        labels = torch.FloatTensor()
        start = True

        # Current author id
        current_author_id = 0

        # For each token
        for token in self.nlp(text_content):
            # Tag SFGRAM
            if u"SFGRAM_START" in token.text:
                # Author
                current_author_id = self.tag2id[token.text[13:]] + 1
            elif u"SFGRAM_STOP" in token.text:
                # Author
                current_author_id = 0
            # end if

            # Vector
            author_vector = torch.zeros(1, self.n_authors + 1)
            author_vector[0, current_author_id] = 1.0

            # Add
            if start:
                labels = author_vector
                start = False
            else:
                labels = torch.cat((labels, author_vector), dim=0)
            # end if
        # end for

        return labels
    # end tag_text

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

        # Transform
        if self.transform is not None:
            # Transform
            transformed = self.transform(text_content)
            text_to_tags = self.tag_text(text_content)
            return transformed, text_to_tags
        else:
            return text_content, self.tag_text(text_content)
        # end if
    # end __getitem__

    ##############################################
    # PRIVATE
    ##############################################

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
            if ".txt" in file_name and f_index < self.dataset_size:
                self.texts.append(os.path.join(self.root, file_name))
            # end if
        # end for

        # Shuffle texts
        shuffle(self.texts)
    # end _load

# end ReutersC50Dataset
