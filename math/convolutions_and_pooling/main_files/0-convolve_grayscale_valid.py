#!/usr/bin/env python3
"""Valid Convolution"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
  """
  A  function that performs a valid convolution on grayscale images
  images is a numpy.ndarray with shape (m, h, w) 
      m is the number of images
      h is the height in pixels of the images
      w is the width in pixels of the images
  kernel is a numpy.ndarray with shape (kh, kw) 
      kh is the height of the kernel
      kw is the width of the kernel
  """
  m = images.shape[0]
  h = images.shape[1]
  w = images.shape[2]
  kh = kernel.shape[0]
  kw = kernel.shape[1]
  output_h = h - kh + 1
  output_w = w - kw + 1

  conv_out = np.zeros((m, output_h, output_w))

  image = np.arange(m)
    # Loop every pixel of the output
  for x in range(output_h):
      for y in range(output_w):
            # element wise multiplication of the kernel and the image
          conv_out[image, x, y] = (np.sum(images[image, x:kh+x,
                                        y:kw+y] * kernel, axis=(1, 2)))
  return conv_out
