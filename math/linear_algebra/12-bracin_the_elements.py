#!/usr/bin/env python3
""" function performs element-wise operations on two matrices """


def np_elementwise(mat1, mat2):
    """
    that performs element-wise addition, subtraction, multiplication, and division

    returns: tuple solutions
    """
    result = []
    result.append(mat1 + mat2)
    result.append(mat1 - mat2)
    result.append(mat1 * mat2)
    result.append(mat1 / mat2)
    return tuple(result)
