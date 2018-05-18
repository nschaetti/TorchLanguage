# -*- coding: utf-8 -*-
#

# Imports
import torch
import math
from decimal import Decimal
import numpy as np


# Perplexity
def perplexity(output_probs, targets, log=False):
    """
    Perplexity
    :param output_probs: Output probabilities for each word/tokens (length x n_tokens)
    :param targets: Real word index
    :return: Perplexity
    """
    pp = Decimal(1.0)
    e_vec = torch.FloatTensor(output_probs.size(0), output_probs.size(1)).fill_(np.e)
    if log:
        set_p = 1.0 / torch.gather(torch.pow(e_vec, exponent=output_probs.data.cpu()), 1,
                                   targets.data.cpu().unsqueeze(1))
    else:
        set_p = 1.0 / torch.gather(output_probs.data.cpu(), 1, targets.data.cpu().unsqueeze(1))
    # end if
    for j in range(set_p.size(0)):
        pp *= Decimal(set_p[j][0])
    # end for
    return pp
# end perplexity


# Cumulative perplexity
def cumperplexity(output_probs, targets, log=False):
    """
    Cumulative perplexity
    :param output_probs:
    :param targets:
    :param log:
    :return:
    """
    # Get prob of test events
    set_p = torch.gather(output_probs, 1, targets.unsqueeze(1))

    # Make sure it's log
    if not log:
        set_p = torch.log(set_p)
    # end if

    # Log2
    set_log = set_p / np.log(2)

    # sum log
    sum_log = torch.sum(set_log)

    # Return
    return sum_log
# end cumperplexity
