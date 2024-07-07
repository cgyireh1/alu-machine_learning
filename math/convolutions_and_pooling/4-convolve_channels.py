#!/usr/bin/env python3
"""Convolution with Channels"""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on images with channels
    images is a numpy.ndarray with shape (m, h, w, c)
    -m is the number of images
    -h is the height in pixels of the images
    -w is the width in pixels of the images
    -c is the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw, c)
    -kh is the height of the kernel
    -kw is the width of the kernel
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
    if ‘same’, performs a same convolution
    if ‘valid’, performs a valid convolution
    if a tuple:
    -ph is the padding for the height of the image
    -pw is the padding for the width of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
    -sh is the stride for the height of the image
    -sw is the stride for the width of the image
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]
    if padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)
    convoluted = np.zeros((m, ((h + (2 * ph) - kh) // sh) + 1,
                           ((w + (2 * pw) - kw) // sw) + 1))
    i = 0
    for h in range(0, (h + (2 * ph) - kh + 1), sh):
        j = 0
        for w in range(0, (w + (2 * pw) - kw + 1), sw):
            convoluted[:, i, j] = np.sum(images[:, h: h + kh, w: w + kw, :] *
                                         kernel, axis=(1, 2, 3))

            j += 1
        i += 1
    return convoluted
