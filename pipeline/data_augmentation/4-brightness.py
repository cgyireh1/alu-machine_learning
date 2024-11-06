#!/usr/bin/env python3
""" brightness of an image """
import tensorflow as tf


def change_brightness(image, max_delta):
    """ Randomly changes the brightness of an image """
    bright = tf.image.random_brightness(image, max_delta)
    return bright
