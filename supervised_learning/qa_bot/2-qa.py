#!/usr/bin/env python3
"""
Task 2: Interactive QA loop with cleaner fallback response.
"""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Starts an interactive loop asking questions about the reference text.

    Args:
        reference (str): Reference document to answer questions from.
    """
    while True:
        question = input("Q: ").strip()

        # Handle exit case (case-insensitive)
        if question.lower() in ("exit", "quit"):
            print("A: Goodbye")
            break

        # Get answer from QA model
        answer = question_answer(question, reference)

        # Respond with appropriate message
        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
