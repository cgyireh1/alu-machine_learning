#!/usr/bin/env python3
"""Unigram BLEU score"""

import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
    Parameters:
        references: is a list of reference translations
        - each reference is a list of the words in the translation
        sentence: is a list containing the model proposed sentence
    Returns: unigram BLEU score
    """
    sen = list(set(sentence))
    count_dict = {}

    # Count appearances
    for reference in references:
        for word in reference:
            if word in sen:
                if word not in count_dict.keys():
                    count_dict[word] = reference.count(word)
                else:
                    new = reference.count(word)
                    old = count_dict[word]
                    count_dict[word] = max(new, old)

    # Clipping
    len_sen = len(sentence)
    list_refs = []
    for reference in references:
        len_ren = len(reference)
        list_refs.append(((abs(len_ren - len_sen)), len_ren))

    # Precision
    ref_len = sorted(list_refs, key=lambda x: x[0])
    ref_len = ref_len[0][1]

    # Penalty
    if len_sen > ref_len:
        bp = 1
    else:
        bp = np.exp(1 - (float(ref_len) / len_sen))

    bleu_score = bp * np.exp(np.log(sum(count_dict.values()) / len_sen))

    return bleu_score
