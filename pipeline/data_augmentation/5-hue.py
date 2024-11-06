#!/usr/bin/env python3
""" hue of an image """
import tensorflow as tf


def change_hue(image, delta):
    """ Changes the hue of an image """
    hue = tf.image.adjust_hue(image, delta)
    return hue
