#!/usr/bin/env python3

import numpy as np


def from_numpy(array):
    """
    function  that creates a pd.DataFrame from a np.ndarray:

    array is the np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order and capitalized. There will not be more than 26 columns.
    Returns: the newly created pd.DataFrame
    """
    c_list = list('ABCDEFGH')
    reshape = c_list[:array.shape[1]]
    return pd.DataFrame(array, columns=reshape)