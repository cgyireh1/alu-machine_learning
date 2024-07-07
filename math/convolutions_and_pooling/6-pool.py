#!/usr/bin/env python3
"""Pooling"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function that performs pooling on images
    images is a numpy.ndarray with shape (m, h, w, c)
    -m is the number of images
    -h is the height in pixels of the images
    -w is the width in pixels of the images
    -c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw)
    -kh is the height of the kernel
    -kw is the width of the kernel
    stride is a tuple of (sh, sw)
    -sh is the stride for the height of the image
    -sw is the stride for the width of the image
    mode indicates the type of pooling
    -max indicates max pooling
    -avg indicates average pooling
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = ((h - kh) // sh) + 1
    pw = ((w - kw) // sw) + 1
    pooled = np.zeros((m, ph, pw, c))
    i = 0
    for h in range(0, (h - kh + 1), sh):
        j = 0
        for w in range(0, (w - kw + 1), sw):
            if mode == 'max':
                pooled[:, i, j, :] = np.max(images[:, h:h + kh, w:w + kw, :],
                                            axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.average(images[:, h:h + kh, w:w + kw, :],
                                                axis=(1, 2))
            else:
                pass
            j += 1
        i += 1
    return pooled