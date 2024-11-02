#!/usr/bin/env python3
""" insert document in python"""


def insert_school(mongo_collection, **kwargs):
    """
    Prototype: def insert_school(mongo_collection, **kwargs):
    mongo_collection will be the pymongo collection object
    Returns the new _id
    """
    return mongo_collection.insert_one(kwargs).inserted_id
