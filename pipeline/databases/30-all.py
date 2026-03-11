#!/usr/bin/env python3
"""List all documents in a MongoDB collection"""


def list_all(mongo_collection):
    """Return a list of all documents in the collection"""
    if mongo_collection is None:
        return []
    return list(mongo_collection.find())
