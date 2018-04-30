# -*- coding: utf-8 -*-
#

# Imports
import numpy as np
from sklearn.decomposition import PCA


# Display a tensor on a 2-dim plot with PCA
def plotPCA(tensor, labels):
    """
    Display a tensor on a 2-dim plot with PCA
    :param tensor: The tensor to display
    :param labels: A dict converting ID to labels
    :return: The image
    """
    # Numpy tensor
    numpy_tensor = tensor.numpy()

    # PCA model
    pca = PCA(n_components=2)

    # Fit
    pca.fit_transform(numpy_tensor)
# end plotPCA


# Display a tensor on a 2-dim plot with T-SNE
def plotTSNE(tensor, labels):
    """
    Display a tensor on a 2-dim plot with T-SNE
    :param tensor:
    :param labels:
    :return:
    """
    pass
# end plotTSNE


# Display a tensor on a tree plot with a specific distance measure
