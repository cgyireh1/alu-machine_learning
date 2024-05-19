#!/usr/bin/env python3
""" function that adds two arrays element-wise """


def add_arrays(arr1, arr2):
    """ returns the sum of two arrays added element-wise """
    if len(arr1) != len(arr2):
        return None
    sum_array = []
    for i in range(len(arr1)):
        sum_array.append(arr1[i] + arr2[i])
    return sum_array
