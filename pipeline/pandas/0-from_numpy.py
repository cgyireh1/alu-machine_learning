#!/usr/bin/env python3

import numpy as np


def from_numpy(array):
    """
    function  that creates a pd.DataFrame from a np.ndarray:

    array is the np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order
    and capitalized. There will not be more than 26 columns.
    Returns: the newly created pd.DataFram
    """
    list_ = list('ABCDEFGH')
    reshape = list_[:array.shape[1]]
    return pd.DataFrame(array, columns=reshape)
