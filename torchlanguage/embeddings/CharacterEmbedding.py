# -*- coding: utf-8 -*-
#

# Imports
import torch
import os
import urllib
import zipfile
import torch.utils.model_zoo as model_zoo


# Embedding url
embedding_url = 'http://www.nilsschaetti.com/embedding/'

# Character embedding
class CharacterEmbedding(object):
    """
    Character embedding
    """

    # Constructor
    def __init__(self, n_gram=1, context=2, dim=10):
        """
        Constructor
        :param n_gram: Gram embedding
        :param context: Context size
        :param dim: Embedding dimension
        """
        # Properties
        self.n_gram = n_gram
        self.context = context
        self.dim = dim
        self.cache_path = os.path.join(os.getenv("HOME"), ".torchlanguage")
        self.embedding_path = os.path.join(self.cache_path, "embedding")

        # Create cache
        self._create_cache()

        # Embedding filename
        embedding_filename = "c" + str(n_gram) + "_cx" + str(context) + "_d" + str(dim) + ".p"
        embedding_zip = "c" + str(n_gram) + "_cx" + str(context) + "_d" + str(dim) + ".zip"

        # Download?
        if not os.path.exists(os.path.join(self.embedding_path, embedding_filename)):
            self._download(embedding_zip)
        # end if

        # Load
        self.gram_to_ix, self.weights = self._load(embedding_filename)
        self.voc_size = len(self.gram_to_ix.keys())
        self.embedding_dim = self.weights.size(1)

        # Compute reverse dictionary
        self.ix_to_gram = self._compute_reverse(self.gram_to_ix)
    # end __init__

    ################################################
    # PRIVATE
    ################################################

    # Download embedding
    def _download(self, embedding_zip):
        """
        Download
        :return:
        """
        # Path to zip file
        path_to_zip = os.path.join(self.embedding_path, embedding_zip)

        # Download
        print(u"Downloading http://www.nilsschaetti.com/embedding/{}".format(embedding_zip))
        urllib.urlretrieve("http://www.nilsschaetti.com/embedding/" + embedding_zip, path_to_zip)

        # Unzip
        zip_ref = zipfile.ZipFile(path_to_zip, 'r')
        zip_ref.extractall(self.embedding_path)
        zip_ref.close()

        # Delete zip
        os.remove(path_to_zip)
    # end _download

    # Create cache
    def _create_cache(self):
        """
        Create cache
        :return:
        """
        # Paths
        models_path = os.path.join(self.cache_path, "models")

        # Check root
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)
        # end if

        # Create sub dirs
        for dir in [self.embedding_path, models_path]:
            if not os .path.exists(dir):
                os.mkdir(dir)
            # end if
        # end for
    # end _create_cache

    # Load embedding
    def _load(self, embedding_filename):
        """
        Load embedding
        :return:
        """
        path_to_p = os.path.join(self.embedding_path, embedding_filename)
        token_to_ix, weights = torch.load(open(path_to_p, 'rb'))
        return token_to_ix, weights
    # end _load

    # Compute reverse
    def _compute_reverse(self, dic):
        """
        Compute reverse
        :param dic:
        :return:
        """
        reverse = dict()
        for key in dic.keys():
            reverse[dic[key]] = key
        # end for

        return reverse
    # end _compute_reverse

# end CharacterEmbedding
