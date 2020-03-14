# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import os
import json
import math
import numpy as np


# SFGram block dataset (pretrained features)
class SFGramBlockDataset(Dataset):
    """
    SFGram dataset (pretrained features)
    """

    # Constructor
    def __init__(self, author, root='./sfgram_block', feature='wv', trained=False, block_length=40, n_files=91,
                 fold=0, k=5, set='train', per_file=False):
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
        self.load_type = self.feature + ("T" if self.trained else "")
        self.trues_count = 0
        self.input_dim = 0
        self.data_dim = 0
        self.data_path = os.path.join(self.author_root, self.load_type)
        self.n_files = n_files
        self.file_start_positions = dict()
        self.fold = fold
        self.k = k
        self.set = set
        self.train_folds, self.dev_folds, self.test_folds, self.train_sizes, self.dev_sizes, self.test_sizes, \
        self.indexes = self._create_folds(self.k)
        self.per_file = per_file
        self.per_file_inputs = list()
        self.per_file_outputs = list()

        # Generate data set
        self._load()
    # end __init__

    #region PUBLIC

    # Next fold
    def next_fold(self):
        """
        Next fold
        :return:
        """
        self.fold += 1
        self._load()
    # end next_fold

    # Set fold
    def set_fold(self, fold):
        """
        Set fold
        :param fold:
        :return:
        """
        self.fold = fold
        self._load()
    # end set_fold

    #endregion PUBLIC

    # region PRIVATE

    # Create folds
    def _create_folds(self, k):
        """
        Create folds
        :param indexes:
        :return:
        """
        # Indexes
        indexes = [x for x in range(self.n_files)]

        # Dataset length
        length = len(indexes)

        # Division and rest
        division = int(math.floor(length / k))
        reste = length - division * k
        reste_size = k - reste

        # Folds size
        fold_sizes = [division + 1] * (reste) + [division] * (reste_size)

        # Folds
        train_folds = list()
        dev_folds = list()
        test_folds = list()

        # Folds size
        train_sizes = list()
        dev_sizes = list()
        test_sizes = list()

        # Fill each folds with indexes
        start = 0
        for i in range(k):
            # Dev and test sizes
            dev_folds_size = int(fold_sizes[i] / 2.0)
            test_folds_size = fold_sizes[i] - dev_folds_size

            # Get dev and test indices
            dev_indices = indexes[start:start+dev_folds_size]
            test_indices = indexes[start+dev_folds_size:start+dev_folds_size+test_folds_size]

            # Append dev and test indices
            dev_folds.append(dev_indices)
            test_folds.append(test_indices)

            # Remove dev and test from total
            indexes_tmp = [value for value in indexes if (value not in dev_indices and value not in test_indices)]

            # Append indices
            train_folds.append(indexes_tmp)

            # Save sizes
            train_sizes.append(len(indexes_tmp))
            dev_sizes.append(dev_folds_size)
            test_sizes.append(test_folds_size)

            # Next start
            start += fold_sizes[i]
        # end for

        return train_folds, dev_folds, test_folds, train_sizes, dev_sizes, test_sizes, indexes
    # end _create_folds

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
        self.data_dim = data_info['tensor_dim']
        self.input_dim = data_info['input_dim']

        # Position in global tensor
        t = 0

        # Target set
        if self.set == "train":
            target_set = self.train_folds[self.fold]
        elif self.set == "dev":
            target_set = self.dev_folds[self.fold]
        else:
            target_set = self.test_folds[self.fold]
        # end if

        # Per file mode
        if self.per_file:
            # Init. data
            self.per_file_inputs = list()
            self.per_file_outputs = list()

            # Total length
            self.data_length = 0
            self.trues_count = 0
            self.dataset_size = 0

            # For each file to load
            for i, data_file_i in enumerate(target_set):
                # Path to the data file
                data_file_path = os.path.join(self.data_path, "sfgram.{}.p".format(data_file_i))

                # Load file with torch
                (document_inputs, document_outputs) = torch.load(data_file_path)

                # Append
                self.per_file_inputs.append(document_inputs)
                self.per_file_outputs.append(document_outputs)

                # Add
                self.data_length += document_inputs.size(0)
                self.trues_count += int(torch.sum(document_outputs).item())
                self.dataset_size += 1
            # end for
        else:
            # Init. data
            self.data_inputs = None
            self.data_outputs = None

            # For each file to load
            for i, data_file_i in enumerate(target_set):
                # Path to the data file
                data_file_path = os.path.join(self.data_path, "sfgram.{}.p".format(data_file_i))

                # Load file with torch
                (document_inputs, document_outputs) = torch.load(data_file_path)

                # Append input and output
                if i == 0:
                    self.data_inputs = document_inputs
                    self.data_outputs = document_outputs
                else:
                    self.data_inputs = torch.cat((self.data_inputs, document_inputs), dim=0)
                    self.data_outputs = torch.cat((self.data_outputs, document_outputs), dim=0)
                # end if

                # Save starting position
                self.file_start_positions[t] = data_file_i

                # Position
                t += document_inputs.size(0)
            # end for

            # Dataset info
            self.data_length = self.data_inputs.size(0)
            self.dataset_size = int(self.data_length / self.block_length)
            self.trues_count = int(torch.sum(self.data_outputs).item())
        # end if
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
        # Per file mode
        if self.per_file:
            # Global true
            if torch.max(self.per_file_outputs[idx]).item() == 1.0:
                global_true = torch.LongTensor([1])
            else:
                global_true = torch.LongTensor([0])
            # end if
            return self.per_file_inputs[idx], self.per_file_outputs[idx], global_true
        else:
            # Position
            pos_start = idx * self.block_length
            pos_end = pos_start + self.block_length

            # Global true
            if torch.max(self.data_outputs[pos_start:pos_end]).item() == 1.0:
                global_true = torch.LongTensor([1])
            else:
                global_true = torch.LongTensor([0])
            # end if

            # Return
            return self.data_inputs[pos_start:pos_end], self.data_outputs[pos_start:pos_end], global_true
        # end if
    # end __getitem__

    #endregion OVERRIDE

# end SFGramDataset
