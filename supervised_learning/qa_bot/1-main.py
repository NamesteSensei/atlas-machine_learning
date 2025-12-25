#!/usr/bin/env python3

answer_loop = __import__('1-loop').answer_loop

if __name__ == "__main__":
    with open("ZendeskArticles/PeerLearningDays.md") as f:
        reference = f.read()

    answer_loop(reference)
