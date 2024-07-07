#!/usr/bin/env python3
"""
Strided Convolution
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w)
    m is the number of images
    h is the  h in pixels of the images
    w is the w in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    kh is the  h of the kernel
    kw is the w of the kernel
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
    if ‘same’, performs a same convolution
    if ‘valid’, performs a valid convolution
    if a tuple:
    ph is the padding for the  h of the image
    pw is the padding for the w of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
    sh is the stride for the  h of the image
    sw is the stride for the w of the image
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]
    if padding is 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding is 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    img_padding = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                         'constant', constant_values=0)
    out_h = ((h + (2 * ph) - kh) // sh) + 1
    out_w = ((w + (2 * pw) - kw) // sw) + 1
    convolved_img = np.zeros((m, out_h, out_w))
    i = 0
    for h in range(0, (h + (2 * ph) - kh + 1), sh):
        j = 0
        for w in range(0, (w + (2 * pw) - kw + 1), sw):
            convolved_img[:, i, j] = np.sum(img_padding[:, h: h + kh,
                                            w: w + kw] * kernel,
                                            axis=(1,2))
            j += 1
        i += 1
    return convolved_img
