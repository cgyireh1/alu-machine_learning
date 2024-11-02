#!/usr/bin/env python3
"""
Python function that returns all students sorted by average score
"""


def top_students(mongo_collection):
    """
    mongo_collection will be the pymongo collection object
    The top must be ordered
    The average score must be part of each item returns with key = averageScore
    """
    students = []
    documents = mongo_collection.find()
    for student in documents:
        total_score = 0
        topics = student["topics"]
        for project in topics:
            total_score += project["score"]
        average_score = total_score / len(topics)
        student["averageScore"] = average_score
        students.append(student)
    students = sorted(students, key=lambda i: i["averageScore"], reverse=True)
    return students
