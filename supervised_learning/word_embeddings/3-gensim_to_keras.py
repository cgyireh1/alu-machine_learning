#!/usr/bin/env python3
"""Extract Word2Vec"""

def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer:
    - model is a gensim word2vec models
    Returns: the trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
