#!/usr/bin/env python3
"""
Function that calculates the adjugate matrix of a matrix
"""


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix
     matrix [list of lists]:else
     Return: the adjugate matrix of matrix
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    height = len(matrix)
    if height is 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != height:
            raise ValueError("matrix must be a non-empty square matrix")
    if height is 1:
        return [[1]]
    multiplier = 1
    cofactor_matrix = []
    for row_i in range(height):
        cofactor_row = []
        for column_i in range(height):
            sub_matrix = []
            for row in range(height):
                if row == row_i:
                    continue
                new_row = []
                for column in range(height):
                    if column == column_i:
                        continue
                    new_row.append(matrix[row][column])
                sub_matrix.append(new_row)
            cofactor_row.append(multiplier * determinant(sub_matrix))
            multiplier *= -1
        cofactor_matrix.append(cofactor_row)
        if height % 2 is 0:
            multiplier *= -1
    adjugate = transpose(cofactor_matrix)
    return adjugate


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    matrix [list of lists]: else
    Return the determinant of the matrix.
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    height = len(matrix)
    if height is 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) is 0 and height is 1:
            return 1
        if len(row) != height:
            raise ValueError("matrix must be a square matrix")
    if height is 1:
        return matrix[0][0]
    if height is 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return ((a * d) - (b * c))
    multiplier = 1
    d = 0
    for i in range(height):
        element = matrix[0][i]
        sub_matrix = []
        for row in range(height):
            if row == 0:
                continue
            new_row = []
            for column in range(height):
                if column == i:
                    continue
                new_row.append(matrix[row][column])
            sub_matrix.append(new_row)
        d += (element * multiplier * determinant(sub_matrix))
        multiplier *= -1
    return (d)


def transpose(matrix):
    """
    Calculates the transpose of a square matrix
     matrix [list of lists]:else
     Return: the transpose of the matrix
    """
    height = len(matrix)
    transpose_matrix = []
    for i in range(height):
        t_row = []
        for row in range(height):
            for column in range(height):
                if column == i:
                    t_row.append(matrix[row][column])
        transpose_matrix.append(t_row)
    return transpose_matrix
