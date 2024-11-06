#!/usr/bin/env python3
""" crop image """
import tensorflow as tf


def crop_image(image, size):
    """ Performs a random crop of an image """
    crop = tf.image.random_crop(image, size=size)
    return crop
