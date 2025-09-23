#!/usr/bin/env python3
"""
2-main.py
---------
Test file for Task 2: Adam optimizer configuration.
"""

import tensorflow as tf

build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model

if __name__ == '__main__':
    # Build model (from Task 1)
    model = build_model(
        784,
        [256, 256, 10],
        ['tanh', 'tanh', 'softmax'],
        0.001,
        0.95
    )

    # Optimize model with Adam
    optimize_model(model, 0.01, 0.99, 0.9)

    # Check that model was compiled properly
    print(model.loss)  # Expect: categorical_crossentropy
    opt = model.optimizer
    print(opt.__class__)  # Expect Adam optimizer
    print((opt.lr.numpy(), opt.beta_1, opt.beta_2))  # Hyperparameters
