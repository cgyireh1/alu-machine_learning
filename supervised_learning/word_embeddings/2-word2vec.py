#!/usr/bin/env python3
"""Train Word2Vec"""

import gensim


def word2vec_model(sentences, size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """
    Creates and trains a gensim word2vec model:
    parameters:
        sentences:
            list of sentences to be trained on
        size:
            dimensionality of the embedding layer
        min_count:
            minimum number of occurances of a word for use in training
        window:
            maximum distance between the current and predicted word
                within a sentence
        negative
            size of negative sampling
        cbow:
            determines the training type
            True: CBOW
            False: Skip-gram
        iterations:
            number of iterations to train over
        seed:
            seed for the random number generator
        workers:
            number of worker threads to train the model
    returns:
        the trained model
    """
    model = gensim.models.Word2Vec(sentences, min_count=min_count,
                                   iter=iterations, size=size,
                                   window=window, negative=negative,
                                   seed=seed, sg=cbow, workers=workers)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.iter)

    return model