#!/usr/bin/env python3
import numpy as np

# Import function from "0-create_confusion.py"
create_confusion_matrix = __import__('0-create_confusion').create_confusion_matrix

if __name__ == '__main__':
    # Load provided labels and logits
    lib = np.load('labels_logits.npz')
    labels = lib['labels']
    logits = lib['logits']

    # Compute confusion matrix
    np.set_printoptions(suppress=True)
    confusion = create_confusion_matrix(labels, logits)

    # Print the confusion matrix
    print(confusion)

    # Save the confusion matrix as a compressed file
    np.savez_compressed('confusion.npz', confusion=confusion)
