#!/usr/bin/env python3
"""Cumulative N-gram BLEU score"""

import numpy as np


def ngram(sentence, n):
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
    lst_gram = []
    for i in range(len(sentence)):
        first = i + n
        last = i
        if first >= len(sentence) + 1:
            break
        aux = sentence[last: first]
        result = ' '.join(aux)
        lst_gram.append(result)
    return lst_gram


def ngram_bleu(references, sentence, n):
    """
    ngram_bleu
    Returns: the n-gram BLEU score
    """
    count_dict = {}
    c_grams = ngram(sentence, n)
    c_grams = list(set(c_grams))
    len_trans = len(c_grams)

    # getting grams references
    ref_grams = []
    for reference in references:
        list_grams = ngram(reference, n)
        ref_grams.append(list_grams)

    for grams in ref_grams:
        for word in grams:
            if word in c_grams:
                if word not in count_dict.keys():
                    count_dict[word] = grams.count(word)
                else:
                    curr = grams.count(word)
                    prev = count_dict[word]
                    count_dict[word] = max(curr, prev)

    precision = sum(count_dict.values()) / len_trans
    return precision


def cumulative_bleu(references, sentence, n):
    """
    Cumulative bleu
    Returns: Cumulative bleu
    """
    precision_values = []
    for i in range(1, n+1):
        result = ngram_bleu(references, sentence, i)
        precision_values.append(result)

    best_match_lst = []
    for reference in references:
        ref_len = len(reference)
        diff = abs(ref_len - len(sentence))
        best_match_lst.append((diff, ref_len))

    arranged_list = sorted(best_match_lst, key=lambda x: x[0])
    best_match = arranged_list[0][1]

    # Brevity penalty
    if len(sentence) > best_match:
        bp = 1
    else:
        bp = np.exp(1 - (float(best_match) / len(sentence)))

    Bleu_score = bp * np.exp(np.sum(np.log(precision_values)) / n)
    return Bleu_score
