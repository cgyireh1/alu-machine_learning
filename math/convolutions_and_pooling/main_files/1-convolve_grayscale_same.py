#!/usr/bin/env python3
"""
Same Convolution
"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    function that performs a same convolution on grayscale images
    images is a numpy.ndarray with shape (m, h, w)
    -m is the number of images
    -h is the height in pixels of the images
    -w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    -kh is the height of the kernel
    -kw is the width of the kernel
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = (kh - 1) / 2
    pw = (kw - 1) / 2

    img_padding = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                         mode='constant', constant_values=0)
    convolved_img = np.zeros((m, h, w))

    image = np.arange(m)
    for x in range(h):
        for y in range(w):
            convolved_img[image, x, y] = (np.sum(img_padding[image,
                                                 x:kh+x, y:kw+y] * kernel,
                                                 axis=(1, 2))
    return convolved_img
