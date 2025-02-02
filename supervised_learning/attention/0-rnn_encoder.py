#!/usr/bin/env python3
""" RNN encoder"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    A class  that inherits from tensorflow.keras.layers.Layer
    to encode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        - vocab is an integer representing the size of the
        input vocabulary
        - embedding is an integer representing the
        dimensionality
        of the embedding vector
        - units is an integer representing the number
        of hidden units in the RNN cell
        - batch is an integer representing the batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Returns: a tensor of shape (batch, units)containing
        the initialized hidden states
        """
        initializer = tf.keras.initializers.Zeros()
        # Q=hidden state matrix
        hidden_Q = initializer(shape=(self.batch, self.units))
        return hidden_Q

    def call(self, x, initial):
        """
        x: Tensor of shape (batch, input_seq_len) containing the input
              to the encoder layer
              as word indices within the vocabulary
        initial: tensor of shape (batch, units) containing the initial
                     hidden state
        Returns: outputs, hidden
          Outputs: Tensor of shape (batch, input_seq_len, units)
                    containing the outputs of the encoder
          Hidden: Tensor of shape (batch, units) containing the last
                  hidden state of the encoder
        """
        embedding = self.embedding(x)
        outputs, last_hiddenQ = self.gru(embedding,
                                         initial_state=initial)
        return outputs, last_hiddenQ
