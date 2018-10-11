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
import random
import pickle
from datetime import datetime
import spacy


# Reuters C50 sentence dataset
class ReutersC50SentenceDataset(Dataset):
    """
    Reuters C50 sentence dataset
    """

    # Constructor
    def __init__(self, root='./data', download=False, n_authors=50, dataset_size=100, dataset_start=0, authors=None,
                 transform=None, retain_transform=False, load_transform=False, language='en'):
        """
        Constructor
        :param root: Data root directory.
        :param download: Download the dataset?
        :param n_authors: How many authors from the dataset to load (2 to 50).
        :param dataset_size: How many samples from each author to load (1 to 100).
        :param authors: The list of authors name to load.
        :param transform: A TextTransformer object to apply.
        :param retain_transform:
        """
        # Properties
        self.root = root
        self.n_authors = n_authors if authors is None else len(authors)
        self.dataset_size = dataset_size
        self.dataset_start = dataset_start
        self.authors = authors
        self.transform = transform
        self.author2id = dict()
        self.id2author = dict()
        self.texts = list()
        self.sentences = list()
        self.multi_author_texts = list()
        self.retain_transform = retain_transform
        self.load_transform = load_transform

        # Load spacy model
        self.nlp = spacy.load(name=language)

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

    # Set start
    def set_start(self, start):
        """
        Set start
        :param start:
        :return:
        """
        self.dataset_start = start
    # end set_start

    #############################################
    # OVERRIDE
    #############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.multi_author_texts)
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # Load sentences
        self._load_sentences(self.multi_author_texts[idx])

        # Samples
        samples = list()

        # For each sentence
        for s in self.sentences:
            # Text and author
            text_content, author_name = s

            # Longer than 1
            if len(text_content) > 20:
                # Transform
                if self.transform is not None:
                    # Transform if not found
                    transformed = self.transform(text_content)

                    # No zero dim
                    if type(transformed) is not list and transformed.dim() == 0:
                        continue
                    # end if

                    # Transformed size
                    if type(transformed) is list:
                        transformed_size = len(transformed)
                    elif type(transformed) is torch.LongTensor or type(transformed) is torch.FloatTensor \
                            or type(transformed) is torch.cuda.LongTensor or type(transformed) is torch.cuda.FloatTensor:
                        transformed_size = transformed.size(0)
                    # end if

                    samples.append((transformed, self.author2id[author_name], self._create_labels(author_name, transformed_size), author_name))
                else:
                    samples.append((text_content, self.author2id[author_name], author_name))
                # end if
            # end if
        # end for

        return samples
    # end __getitem__

    ##############################################
    # PRIVATE
    ##############################################

    # Parse text
    def _parse_text(self, author_info):
        """
        Parse text
        :param author_info:
        :return:
        """
        # Texts and authors
        text_path, author_name = author_info

        # Read text
        text_content = codecs.open(text_path, 'r', encoding='utf-8').read()

        # Get sentences
        tokens = self.nlp(text_content)

        # Add each sentence
        for s in tokens.sents:
            current_sentence = unicode(s)
            self.sentences.append((current_sentence, author_name))
        # end for
    # end _parse_text

    # Load sentences
    def _load_sentences(self, authors_info):
        """
        Load sentences
        :param author1:
        :param author2:
        :return:
        """
        # Empty
        self.sentences = list()

        # Parse text 1
        self._parse_text(authors_info[0])

        # Parse text 2
        self._parse_text(authors_info[1])
    # end _load_sentences

    # Save transform
    def _save_transform(self, transform, text_path, transform_name):
        """
        Save transform
        :param text_path:
        :param transform_name:
        :return:
        """
        print(u"load")
        return pickle.dump(transform, open(text_path + "." + transform_name, 'wb'))
    # end _save_transform

    # Load transform
    def _load_transform(self, text_path, transform_name):
        """
        Load transform
        :param text_path:
        :return:
        """
        print(u"load")
        try:
            return pickle.load(open(text_path + "." + transform_name, 'rb'))
        except IOError:
            return None
        # end try
    # end if

    # Create labels
    def _create_labels(self, author_name, length):
        """
        Create labels
        :param author_name:
        :param length:
        :return:
        """
        # Author id
        author_id = self.author2id[author_name]

        # Vector
        tag_vector = torch.zeros(length, self.n_authors)

        # Set
        tag_vector[:, author_id] = 1.0

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
        path_to_zip = os.path.join(self.root, "reutersc50.zip")

        # Download
        urllib.urlretrieve("http://www.nilsschaetti.com/datasets/reutersc50.zip", path_to_zip)

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
        authors_info = json.load(open(os.path.join(self.root, "authors.json"), 'r'))

        # Author count
        author_count = 0

        # Given authors
        if self.authors is not None:
            given_authors = list(self.authors)
        else:
            given_authors = None
        # end if
        self.authors = list()

        # For each authors
        for index, author_name in enumerate(authors_info.keys()):
            # If in the set
            if author_count < self.n_authors and (given_authors is None or author_name in given_authors):
                # New author
                self.author2id[author_name] = author_count
                self.id2author[index] = author_name

                # Add each text
                for text_index, text_name in enumerate(authors_info[author_name]):
                    if text_index >= self.dataset_start and text_index < self.dataset_start + self.dataset_size:
                        # Add
                        self.texts.append((os.path.join(self.root, text_name + ".txt"), author_name))
                    # end if
                # end for

                # Count
                self.authors.append(author_name)
                author_count += 1
            # end if
        # end for

        # Shuffle but always the same
        random.seed(1985)
        random.shuffle(self.texts)

        # Make two authors
        for i in range(len(self.texts)-1):
            self.multi_author_texts.append((self.texts[i], self.texts[i+1]))
        # end for
    # end _load

# end ReutersC50SentenceDataset
