#!/usr/bin/env python3
"""Encode Tokens"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """
    - pt is the tf.Tensor containing the Portuguese sentence
    - en is the tf.Tensor containing the corresponding English sentence
    - The tokenized sentences should include the start and end of sentence tokens
    - The start token should be indexed as vocab_size
    - The end token should be indexed as vocab_size + 1
    Returns: pt_tokens, en_tokens
      - pt_tokens is a np.ndarray containing the Portuguese tokens
      - en_tokens is a np.ndarray. containing the English tokens
    """

    def __init__(self):
        """Class constructor"""
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train, self.data_valid = examples['train'], \
            examples['validation']

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """tokenize data """

        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data),
            target_vocab_size=2 ** 15)

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data),
            target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ encoding """

        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return lang1, lang2
