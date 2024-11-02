#!/usr/bin/env python3
"""
script that provides some stats about Nginx logs stored in MongoDB
"""


from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_collection = client.logs.nginx
    doc_count = logs_collection.count_documents({})
    print("{} logs".format(doc_count))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        method_count = logs_collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, method_count))
    path = {"method": "GET", "path": "/status"}
    path_count = logs_collection.count_documents(path)
    print("{} status check".format(path_count))
