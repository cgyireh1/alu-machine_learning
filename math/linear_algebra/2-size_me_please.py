#!/usr/bin/env python3
""" function calculating the shape of a matrix """


def matrix_shape(matrix):
    """ returning list of integers representing dimensions of the given matrix """
    matrix_shape = []
    while (type(matrix) is list):
        matrix_shape.append(len(matrix))
        matrix = matrix[0]
    return matrix_shape
