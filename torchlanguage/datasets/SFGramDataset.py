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


# SFGram dataset
class SFGramDataset(Dataset):
    """
    SFGram dataset
    """

    # Constructor
    def __init__(self, author, root='./data', download=False, transform=None, load_type='wv', remove_texts=None,
                 homogeneous=True, stream=False, block_length=40, diskblock_length=1024):
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
        self.last_text = ""
        self.remove_texts = remove_texts
        self.homogenous = homogeneous
        self.text2author = dict()
        self.stream = stream
        self.block_length = block_length
        self.diskblock_length = diskblock_length

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

    #region PUBLIC

    # Precompute documents in the dataset
    def precompute_documents(self, target_path):
        """
        Precompute documents in the dataset
        :param target_path: Target directory where to save transformed text
        """
        # Positions
        pos = 0

        # Number of files
        n_files = 0

        # List of positions
        list_of_positions = dict()

        # Vocabulary size
        voc_size = 0

        # Total token from the author
        total_trues = 0

        # Files info
        files_info = dict()

        # Author target path
        author_target_path = os.path.join(target_path, self.author)

        # Create if it does not exists
        if not os.path.exists(author_target_path):
            os.mkdir(author_target_path)
        # end if

        # For each text in the dataset
        for text_i, text_path in enumerate(self.texts):
            # Transform document to inputs and outputs
            document_inputs, document_outputs = self.transform_document(text_path)

            # Add to path
            path_transform = os.path.join(author_target_path, "{}".format(self.load_type))

            # Create if does not exists
            if not os.path.exists(path_transform):
                os.mkdir(path_transform)
            # end if

            # Add file
            path_transform = os.path.join(
                path_transform,
                "sfgram.{}.p".format(text_i)
            )

            # Apply or load
            torch.save((document_inputs, document_outputs), path_transform)

            # Add to position information
            list_of_positions[pos] = text_i

            # How many token from the author
            author_trues = torch.sum(document_outputs).item()
            total_trues += author_trues

            # File information
            files_info[text_i] = {
                'basename': os.path.basename(text_path),
                'length': document_inputs.size(0),
                'trues': author_trues,
                'i': text_i,
                'path': text_path
            }

            # Voc size
            if torch.max(document_inputs).item() > voc_size:
                voc_size = torch.max(document_inputs).item()
            # end if

            # Next start positions
            pos += document_inputs.size(0)

            # N files
            n_files += 1
        # end for

        # Save information
        json.dump(
            {
                'author': self.author,
                'feature': self.load_type,
                'positions': list_of_positions,
                'length': pos,
                'n_files': n_files,
                'files': files_info,
                'trues': total_trues,
                'voc_size': voc_size,
                'tensor_dim': document_inputs.dim(),
                'tensor_type': 'float' if document_inputs.dim() == 2 else 'long',
                'input_dim': document_inputs.size(-1) if document_inputs.dim() == 2 else 0
            },
            open(os.path.join(author_target_path, "sfgam_info.{}.json".format(self.load_type)), 'w'),
            sort_keys=True,
            indent=4
        )
    # end precompute_documents

    # Precompute documents in the dataset
    # def precompute_documents(self, target_path):
        """
        Precompute documents in the dataset
        :param target_path: Target directory where to save transformed text
        """
        # Number of diskblocks
        """n_diskblocks = 0

        # For each text in the dataset
        for text_path in self.texts:
            # Transform document to inputs and outputs
            document_inputs, document_outputs = self.transform_document(text_path)

            # For each blocks
            for bt in range(0, document_inputs.size(0), self.diskblock_length):
                # Take current blocks
                document_inputs_block = document_inputs[bt:bt+self.diskblock_length]
                document_outputs_block = document_outputs[bt:bt+self.diskblock_length]

                # If end of the document, put zeros
                if document_inputs_block.size(0) < self.diskblock_length:
                    # Depend on type
                    if document_inputs_block.dim() == 2:
                        # Inputs
                        tmp_inputs_block = torch.zeros(document_inputs_block.size(0), document_inputs_block.dim(1))
                        tmp_inputs_block[:document_inputs_block.size(0)] = document_inputs_block

                        # Outputs
                        tmp_outputs_block = torch.zeros(document_outputs_block.size(0))
                        tmp_outputs_block[:document_outputs_block.size(0)] = document_outputs_block
                    else:
                        # Inputs
                        tmp_inputs_block = torch.LongTensor(document_inputs_block.size(0)).fill_(0)
                        tmp_inputs_block[:document_inputs_block.size(0)] = document_inputs_block

                        # Outputs
                        tmp_outputs_block = torch.zeros(document_outputs_block.size(0), 1)
                        tmp_outputs_block[:document_outputs_block.size(0)] = document_outputs_block
                    # end if

                    # Change
                    document_inputs_block = tmp_inputs_block
                    document_outputs_block = tmp_outputs_block
                # end if

                # Add to path
                path_transform = os.path.join(target_path, "{}.{}".format(self.load_type, self.author))

                # Create if does not exists
                if not os.path.exists(path_transform):
                    os.mkdir(path_transform)
                # end if

                # Add file
                path_transform = os.path.join(
                    path_transform,
                    "sfgram_block.{}.p".format(n_diskblocks)
                )

                # Apply or load
                torch.save((document_inputs_block, document_outputs_block), path_transform)

                # One more block
                n_diskblocks += 1
            # end for
        # end for

        # Save information
        json.dump(
            {
                'n_diskblocks': n_diskblocks,
                'total_length': n_diskblocks * self.diskblock_length,
                'feature': self.load_type
            },
            open(os.path.join(target_path, "sfgam_info.{}.{}.json".format(self.load_type, self.author)), 'w')
        )"""
    # end precompute_documents

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

    # Transform length
    def transform_length(self, text_path):
        """
        Get transformed text length
        """
        # Read text
        text_content = codecs.open(text_path, 'r', encoding='utf-8').read()

        # Segment text
        segments = self.segment_text(text_content)

        # Transform
        if self.transform is not None:
            # Path to text information
            path_info = "{}.{}.{}.json".format(text_path, self.load_type, self.author)

            # Apply or load
            if os.path.exists(path_info):
                # Load information from JSON
                text_info = json.loads(codecs.open(path_info, 'r', encoding='utf-8').read())

                # Return length
                return text_info[self.load_type]
            else:
                # Get transformed text
                text_transformed = self.transform(text_content)

                # Transformed text length
                text_transformed_length = text_transformed.size(0)

                # Save information
                json.dump({
                    'path': text_path,
                    'segments': len(segments),
                    self.load_type: text_transformed_length
                }, open(path_info, 'w'))

                return text_transformed_length
            # end if
        # end if

        return 0
    # end transform_length

    # Transform text
    def transform_text(self, text_path, n, text_content, author_prob):
        """
        Transform text
        :return:
        """
        # Transform
        if self.transform is not None:
            # Path to transformation
            path_transform = "{}.{}.{}.{}.p".format(text_path, self.load_type, n, self.author)

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

            # Return
            return transformed, author_prob, self._create_labels(author_prob, transformed_size)
        else:
            # Return
            return text_content, author_prob
        # end if
    # end transform_text

    # Transform a document into inputs and outputs
    def transform_document(self, text_path):
        """
        Transform a document into inputs and outputs
        :param text_path: Path to text file
        """
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
                    segment_text = segment_text.replace(t, "")
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
    # end transform_document

    #endregion PUBLIC

    # region PRIVATE

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
        urllib.request.urlretrieve("http://www.nilsschaetti.com/datasets/sfgram.zip", path_to_zip)

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

    # endregion PRIVATE

    #region OVERRIDE

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
                    segment_text = segment_text.replace(t, "")
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

    #endregion OVERRIDE

# end SFGramDataset
