#!/usr/bin/env python3
"""
Task 1: Interactive QA loop using question_answer().
"""

question_answer = __import__('0-qa').question_answer


def answer_loop(ref):
    """
    Starts an interactive loop asking questions about the reference text.

    Args:
        ref (str): Reference document to answer questions from.
    """
    while True:
        question = input("Q: ").strip()

        if question.lower() in ("exit", "quit"):
            break

        answer = question_answer(question, ref)
        print("A: {}".format(answer if answer else "Sorry, I do not know."))
