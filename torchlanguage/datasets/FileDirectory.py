# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import urllib
import zipfile
import os
import codecs
import random


# File directory dataset
class FileDirectory(Dataset):
    """
    Load files from a directory
    """

    # Constructor
    def __init__(self, root='./data', download=False, transform=None, download_url="", save_transform=False, shuffle=False):
        """
        Constructor
        :param root: Data root directory.
        :param download: Download the dataset?
        :param transform: A TextTransformer object to apply.
        """
        # Properties
        self.root = root
        self.transform = transform
        self.download_url = download_url
        self.save_transform = save_transform
        self.shuffle = shuffle

        # Create directory if needed
        if not os.path.exists(self.root):
            self._create_root()
        # end if

        # Download the data set
        if download and download_url != "":
            self._download()
        # end if

        # List file
        self.samples, self.classes = self._load()

        # Shuffle
        if shuffle:
            random.seed(1)
            random.shuffle(self.samples)
        # end if

        # N classes
        self.n_classes = len(self.classes)
    # end __init__

    #############################################
    # PUBLIC
    #############################################

    #############################################
    # PRIVATE
    #############################################

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
        path_to_zip = os.path.join(self.root, "dataset.zip")

        # Download
        urllib.urlretrieve(self.download_url, path_to_zip)

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
        Load dataset
        :return:
        """
        samples = list()
        classes = list()
        for file_name in os.listdir(self.root):
            if file_name[-4:] == ".txt":
                # Get class name
                class_name = unicode(file_name[:file_name.find("_")])
                title = unicode(file_name[file_name.find("_")+1:-4])
                samples.append((file_name, class_name, title))
                if class_name not in classes:
                    classes.append(class_name)
                # end if
            # end if
        # end for
        return samples, classes
    # end _load

    #############################################
    # OVERRIDE
    #############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.samples)
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # Truth
        file_name, class_name, title = self.samples[idx]

        # Transformation
        if self.transform is not None:
            if os.path.exists(os.path.join(self.root, file_name + ".pth")):
                return torch.load(open(os.path.join(self.root, file_name + ".pth"), "rb")), class_name, title
            else:
                try:
                    transformed = self.transform(codecs.open(os.path.join(self.root, file_name), 'r', encoding='utf-8').read())
                except UnicodeDecodeError:
                    transformed = self.transform(codecs.open(os.path.join(self.root, file_name), 'r', encoding='windows-1252').read())
                # end try
                # transformed = self.transform(unicode(codecs.open(os.path.join(self.root, file_name), 'rb').read()))
                if self.save_transform:
                    torch.save(transformed, open(os.path.join(self.root, file_name + ".pth"), "wb"))
                # end if
                return transformed, class_name, title
            # end if
        else:
            try:
                return codecs.open(os.path.join(self.root, file_name), 'r', encoding='utf-8').read(), class_name, title
            except UnicodeDecodeError:
                return codecs.open(os.path.join(self.root, file_name), 'r', encoding='windows-1252').read(), class_name, title
            # end if
        # end if
    # end __getitem__

# end FileDirectory
