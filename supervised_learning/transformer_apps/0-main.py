#!/usr/bin/env python3
"""Main file to test Dataset class for TED translation dataset"""

Dataset = __import__('0-dataset').Dataset

def main():
    """
    Runs sample tests on the Dataset class to confirm loading and tokenizers.
    """
    data = Dataset()

    # Print one sample from training set
    for pt, en in data.data_train.take(1):
        print(pt.numpy().decode('utf-8'))
        print(en.numpy().decode('utf-8'))

    # Print one sample from validation set
    for pt, en in data.data_valid.take(1):
        print(pt.numpy().decode('utf-8'))
        print(en.numpy().decode('utf-8'))

    # Print tokenizer types
    print(type(data.tokenizer_pt))
    print(type(data.tokenizer_en))


if __name__ == "__main__":
    main()
