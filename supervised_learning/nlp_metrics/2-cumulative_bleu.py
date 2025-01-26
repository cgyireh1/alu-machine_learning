#!/usr/bin/env python3
"""Cumulative N-gram BLEU score"""

import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence:
    Parameters:
    - references: is a list of reference translations
    --each reference translation is a list of the words in the trans
    - sentence is a list containing the model proposed sentence
    - n is the size of the largest n-gram to use for evaluation
    - All n-gram scores should be weighted evenly
    Returns:
    The cumulative n-gram BLEU score
    """
    if n == 1:
        return references, sentence

    ngram_sentence = []
    sentence_length = len(sentence)

    for i, word in enumerate(sentence):
        count = 0
        w = word
        for j in range(1, n):
            if sentence_length > i + j:
                w += " " + sentence[i + j]
                count += 1
        if count == j:
            ngram_sentence.append(w)

    ngram_references = []

    for ref in references:
        ngram_ref = []
        ref_length = len(ref)

        for i, word in enumerate(ref):
            count = 0
            w = word
            for j in range(1, n):
                if ref_length > i + j:
                    w += " " + ref[i + j]
                    count += 1
            if count == j:
                ngram_ref.append(w)
        ngram_references.append(ngram_ref)

    return ngram_references, ngram_sentence


def precision(references, sentence, n):
    """
    Calculates the precision for n-gram BLEU score for a sentence
    returns:
        the precision for n-gram BLEU score
    """
    ngram_references, ngram_sentence = transform_grams(references, sentence, n)
    ngram_sentence_length = len(ngram_sentence)
    sentence_length = len(sentence)

    sentence_dictionary = {word: ngram_sentence.count(word) for
                           word in ngram_sentence}
    references_dictionary = {}

    for ref in ngram_references:
        for gram in ref:
            if references_dictionary.get(gram) is None or \
               references_dictionary[gram] < ref.count(gram):
                references_dictionary[gram] = ref.count(gram)

    matchings = {word: 0 for word in ngram_sentence}

    for ref in ngram_references:
        for gram in matchings.keys():
            if gram in ref:
                matchings[gram] = sentence_dictionary[gram]

    for gram in matchings.keys():
        if references_dictionary.get(gram) is not None:
            matchings[gram] = min(references_dictionary[gram], matchings[gram])

    precision = sum(matchings.values()) / ngram_sentence_length

    return precision


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence
    returns:
        the cumulative n-gram BLEU score
    """
    sentence_length = len(sentence)
    precisions = [0] * n

    for i in range(n):
        precisions[i] = precision(references, sentence, i + 1)

    mean = np.exp(np.sum((1 / n) * np.log(precisions)))

    index = np.argmin([abs(len(word) - sentence_length) for
                       word in references])

    references_length = len(references[index])

    if sentence_length > references_length:
        BLEU = 1
    else:
        BLEU = np.exp(1 - float(references_length) / sentence_length)

    BLEU_score = BLEU * mean

    return BLEU_score
