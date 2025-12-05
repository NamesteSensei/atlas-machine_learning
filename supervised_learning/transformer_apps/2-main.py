#!/usr/bin/env python3
"""
Main script to test Dataset class.
"""

Dataset = __import__('2-dataset').Dataset


def main():
    """Instantiate Dataset and print one example from train and valid sets."""
    data = Dataset()
    for pt, en in data.data_train.take(1):
        print(pt, en)
    for pt, en in data.data_valid.take(1):
        print(pt, en)


if __name__ == "__main__":
    main()
