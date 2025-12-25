#!/usr/bin/env python3

semantic_search = __import__('3-semantic_search').semantic_search

if __name__ == "__main__":
    result = semantic_search("ZendeskArticles", "When are PLDs?")
    print(result)
