#!/usr/bin/env python3
"""
4-qa.py: Multi-reference QA using semantic search + transformer QA
"""

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import os


def semantic_search(corpus_path, question):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    docs = []
    contents = []

    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)
        with open(file_path, 'r') as f:
            content = f.read()
            contents.append(content)
            docs.append(model.encode(content, convert_to_tensor=True))

    query_emb = model.encode(question, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_emb, doc)[0][0].item()
              for doc in docs]
    best_match = scores.index(max(scores))
    return contents[best_match]


def question_answer(corpus_path):
    qa = pipeline("question-answering",
                  model="distilbert-base-cased-distilled-squad")

    while True:
        question = input("Q: ").strip()
        if question.lower() in ["exit", "quit", "goodbye"]:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, question)
        result = qa(question=question, context=reference)
        answer = result.get('answer', '').strip()

        if not answer or answer.lower() in ["[cls]", ""]:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
