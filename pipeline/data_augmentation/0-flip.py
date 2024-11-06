#!/usr/bin/env python3
""" 0.Flip"""
import tensorflow as tf


def flip_image(image):
    """flip image"""
    flip = tf.image.flip_left_right(image)
    return flip
