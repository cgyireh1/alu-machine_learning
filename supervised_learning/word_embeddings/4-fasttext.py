#!/usr/bin/env python3
"""
FastText
"""


from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a gensim FastText model
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
    if cbow is True:
        cbow_flag = 0
    else:
        cbow_flag = 1
    model = FastText(sentences=sentences,
                     size=size,
                     min_count=min_count,
                     window=window,
                     negative=negative,
                     sg=cbow_flag,
                     iter=iterations,
                     seed=seed,
                     workers=workers)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    return model