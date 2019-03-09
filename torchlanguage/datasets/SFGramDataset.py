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
import math


# SFGram dataset
class SFGramDataset(Dataset):
    """
    SFGram dataset
    """

    # Constructor
    def __init__(self, author, root='./data', download=False, transform=None, load_type='wv', remove_texts=None, homogeneous=True):
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
        self.last_text = u""
        self.remove_texts = remove_texts
        self.homogenous = homogeneous
        self.text2author = dict()

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
            # print(path_transform)
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

        # Last text
        self.last_text = text_path

        # Read text
        text_content = codecs.open(text_path, 'r', encoding='utf-8').read()

        # Segment text
        segments = self.segment_text(text_content)

        # Transform each segment
        for i, segment in enumerate(segments):
            # Segment
            segment_text, segment_author = segment

            # Remove texts
            if self.remove_texts is not None:
                for t in self.remove_texts:
                    segment_text = segment_text.replace(t, u"")
                # end for
            # end if

            # Ok
            if len(segment_text) > 0:
                if self.transform is not None:
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
                else:
                    # Transform
                    if segment_author == self.author:
                        segment_transformed, segment_labels = self.transform_text(text_path, i, segment_text, 1.0)
                    else:
                        segment_transformed, segment_labels = self.transform_text(text_path, i, segment_text, 0.0)
                    # end if

                    if i == 0:
                        text_transformed = [segment_transformed]
                        text_labels = [int(segment_labels)]
                    else:
                        text_transformed.append(segment_transformed)
                        text_labels.append(int(segment_labels))
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
        author_texts = list()

        # For each author
        for author_index, author_name in enumerate(self.authors_info.keys()):
            self.author2id[author_name] = author_index
            self.id2author[author_index] = author_name
            self.id2tag[author_index] = self.authors_info[author_name]['id']
            self.tag2id[self.authors_info[author_name]['id']] = author_index

            # Text to author
            if self.authors_info[author_name]['id'] == self.author:
                for t in self.authors_info[author_name]['texts']:
                    author_texts.append(t['magazine'])
                # end for
            # end if
        # end for

        # For each text
        for f_index, file_name in enumerate(os.listdir(self.root)):
            text_name = file_name[:-4]
            text_ext = file_name[-4:]
            if text_ext == ".txt":
                text_path = os.path.join(self.root, file_name)
                self.texts.append(text_path)
                if text_name in author_texts:
                    self.text2author[text_path] = self.author
                else:
                    self.text2author[text_path] = 'NONE'
                # end if
            # end if
        # end for

        # Shuffle texts
        seed(1)
        shuffle(self.texts)

        # Make it homogeneous
        if self.homogenous:
            self.texts = self._make_homogenous(self.texts)
        # end if
    # end _load

    # Make homogenous
    def _make_homogenous(self, texts):
        """
        Make homogenous
        :param texts:
        :return:
        """
        # List
        none_list = list()
        author_list = list()
        homogenous_list = list()

        # For each text
        for text in texts:
            if self.text2author[text] == 'NONE':
                none_list.append(text)
            else:
                author_list.append(text)
            # end if
        # end for

        # Padding
        author_length = float(len(author_list))
        none_length = float(len(none_list))
        total_length = len(none_list) + len(author_list)
        padding = int(math.ceil(total_length / author_length))

        # For each text
        none_index = 0
        author_index = 0
        for i in range(total_length):
            if i % padding == 0 and author_index < author_length:
                homogenous_list.append(author_list[author_index])
                author_index += 1
            elif none_index < none_length:
                homogenous_list.append(none_list[none_index])
                none_index += 1
            else:
                homogenous_list.append(author_list[author_index])
                author_index += 1
            # end if
        # end for

        return homogenous_list
    # end _make_homogenous

# end ReutersC50Dataset
