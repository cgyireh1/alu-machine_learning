#!/usr/bin/env python3
"""
Convolution with Padding
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that performs a convolution on grayscale 
    images with custom padding
    images is a numpy.ndarray with shape (m, h, w)
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    kh is the height of the kernel
    kw is the width of the kernel
    padding is a tuple of (ph, pw)
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    the image should be padded with 0’s
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = padding[0]
    pw = padding[1]
    out_h = h + (2 * ph) - kh + 1
    out_w = w + (2 * pw) - kw + 1

    img_padding = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 
                       mode='constant', constant_values=0)
    convolved_img = np.zeros((m, out_h, out_w))

    image = np.arange(m)
    for x in range(out_h):
        for y in range(out_w):
            convolved_img[image, x, y] = (np.sum(img_padding[image,
                                                 x:kh+x, y:kw+y] * kernel))
    return convolved_img