#!/usr/bin/env python3
""" List all documents in Python """


def list_all(mongo_collection):
    """
   Prototype: def list_all(mongo_collection):
   Return an empty list if no document in the collection
    """
    docs = []
    collection = mongo_collection.find()
    for doc in collection:
        docs.append(doc)
    return docs
