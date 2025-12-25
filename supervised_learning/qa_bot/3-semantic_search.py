#!/usr/bin/env python3
"""
Task 3: Semantic Search using Sentence Transformers
"""

import os
from sentence_transformers import SentenceTransformer, util


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search to find the most relevant document for the given sentence.

    Args:
        corpus_path (str): Path to directory containing .md reference documents.
        sentence (str): The sentence to search for.

    Returns:
        str: The content of the most similar document.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    documents = []
    file_paths = []

    # Load and encode documents
    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            path = os.path.join(corpus_path, filename)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(content)
                file_paths.append(path)

    # Encode documents and the input sentence
    doc_embeddings = model.encode(documents, convert_to_tensor=True)
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.cos_sim(sentence_embedding, doc_embeddings)[0]

    # Find best matching document
    best_match_idx = similarities.argmax()

    return documents[best_match_idx]
