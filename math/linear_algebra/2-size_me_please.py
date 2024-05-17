#!/usr/bin/env python3

def matrix_shape(matrix):
    """ returns list of integers representing dimensions of given matrix """
    matrix_shape = []
    while (type(matrix) is list):
        matrix_shape.append(len(matrix))
        matrix = matrix[0]
    return matrix_shape
