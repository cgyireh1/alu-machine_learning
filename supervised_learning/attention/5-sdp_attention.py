#!/usr/bin/env python3
"""  Scaled Dot Product Attention """

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
   Q: tensor with its last two dimensions as (..., seq_len_q, dk)
      containing the query matrix
   K: tensor with its last two dimensions as (..., seq_len_v, dk)
      containing the key matrix
   V: tensor with its last two dimensions as (..., seq_len_v, dv)
      containing the value matrix
   mask: tensor that can be broadcast into
   (..., seq_len_q, seq_len_v)
         containing the optional mask, or defaulted to None
   output, weights
    """

    query = tf.matmul(Q, K, transpose_b=True)
    # scale q
    key_dimension = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_query = query / tf.math.sqrt(key_dimension )

    if mask is not None:
        scaled_query += (mask * -1e9)

    weights = tf.nn.softmax(scaled_query, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights
