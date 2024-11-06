#!/usr/bin/env python3
""" rotate Image """
import tensorflow as tf


def rotate_image(image):
    """ Rotates an image by 90 degrees counter-clockwise """
    rotate = tf.image.rot90(image, k=1)
    return rotate
