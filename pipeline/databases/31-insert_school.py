#!/usr/bin/env python3
"""Insert a new document in MongoDB collection"""


def insert_school(mongo_collection, **kwargs):
    """Insert a new document based on kwargs"""
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
