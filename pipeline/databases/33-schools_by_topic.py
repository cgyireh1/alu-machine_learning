#!/usr/bin/env python3
"""List schools with specific toppic"""


def schools_by_topic(mongo_collection, topic):
    """
    mongo_collection will be the pymongo collection object
    topic (string) will be topic searched
    """
    school = []
    results = mongo_collection.find({"topics": {"$all": [topic]}})
    for result in results:
        school.append(result)
    return school
