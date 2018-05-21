# -*- coding: utf-8 -*-
#

# Imports
import math
import numpy as np
import torch
import scipy.spatial.distance


# Compute distance matrix from weight matrix
def distance_matrix(weights, dim=0, distance_measure='cosine'):
    """
    Compute distance matrix from weight matrix
    :param weights: Weight matrix
    :param dim: Weight dimension
    :param distance_measure: How to measure distance between vectors?
    :return: Distance matrix
    """
    # Number of items
    n_items = weights.size(dim)

    # Distance matrix
    if type(weights) == np.matrix or type(weights) == np.array:
        vector_type = 'numpy'
    else:
        vector_type = 'torch'
    # end if

    # Distance matrix
    if vector_type == 'numpy':
        n_dim = weights.ndim
        distance_matrix = np.zeros((n_items, n_items))
    else:
        n_dim = weights.dim()
        distance_matrix = torch.zeros(n_items, n_items)
    # end if

    # For each combination
    for i in range(n_items):
        for j in range(n_items):
            indexor1 = [None] * n_dim
            indexor2 = [None] * n_dim
            indexor1[dim] = i
            indexor2[dim] = j
            indexor1 = tuple(indexor1)
            indexor2 = tuple(indexor2)

            # Vectors
            if vector_type == 'numpy':
                v = weights[indexor1]
                u = weights[indexor2]
            else:
                v = weights[indexor1].numpy()
                u = weights[indexor2].numpy()
            # end if

            # Distance
            if distance_measure == 'cosine':
                distance_matrix[i, j] = scipy.spatial.distance.cosine(u, v)
            # end if
        # end for
    # end for

    return distance_matrix
# end distance_matrix
