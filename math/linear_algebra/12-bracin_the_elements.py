#!/usr/bin/env python3
""" function performs element-wise operations on two matrices """


def np_elementwise(mat1, mat2):
    """
    returns a tuple of the element-wise sum, difference, product, and quotient
    """
    result = []
    result.append(mat1 + mat2)
    result.append(mat1 - mat2)
    result.append(mat1 * mat2)
    result.append(mat1 / mat2)
    return tuple(result)
