#!/usr/bin/env python3
""" Transformer Decoder Block"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ A class that inherits from
    tensorflow.keras.layers.Layer to create
    the encoder for a transformer """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        dm: the dimensionality of the model
        h: the number of heads
        hidden: the number of hidden units in the fully connected layer
        drop_rate: the dropout rate
        Public instance attributes:
        mha: a MultiHeadAttention layer
        dense_hidden: the hidden dense layer with hidden units and
                        relu activation
        dense_output: the output dense layer with dm units
        layernorm1: the first layer norm layer, with epsilon=1e-6
        layernorm2: the second layer norm layer, with epsilon=1e-6
        layernorm3 - the third layer norm layer, with epsilon=1e-6
        dropout1: the first dropout layer
        dropout2: the second dropout layer
        dropout3 - the third dropout layer
        """
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        x: Tensor of shape (batch, target_seq_len, dm)containing the i
            nput to the decoder block
        encoder_output: tensor of shape (batch, input_seq_len, dm)
                        containing the output of the encoder
        training: boolean to determine if the model is training
        look_ahead_mask: mask to be applied to the first multi head
                          attention layer
        padding_mask: mask to be applied to the second multi head
                      attention layer
        Returns: tensor of shape (batch, target_seq_len, dm)
              containing the block’s output
        """
        # ffn = feed forward neural network
        attention, attention_block = self.mha1(x, x, x, look_ahead_mask)
        attention = self.dropout1(attention, training=training)
        output1 = self.layernorm1(attention + x)
        attention2, attn_weights_block2 = self.mha2(output1,
                                                    encoder_output,
                                                    encoder_output,
                                                    padding_mask)
        attention2 = self.dropout2(attention2, training=training)
        output2 = self.layernorm2(attention2 + output1)
        hidden_output = self.dense_hidden(output2)
        output_output = self.dense_output(hidden_output)
        ffn_output = self.dropout3(output_output, training=training)
        output = self.layernorm3(ffn_output + output2)

        return output
