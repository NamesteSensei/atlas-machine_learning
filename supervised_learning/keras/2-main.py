#!/usr/bin/env python3

import tensorflow.keras as K

build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model

if __name__ == '__main__':
    model = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    optimize_model(model, 0.01, 0.99, 0.9)
    
    print(model.loss)  # Should output 'categorical_crossentropy'
    
    optimizer = model.optimizer
    print(optimizer.__class__)  # Should output the Adam optimizer class
    
    # Correctly accessing learning rate and beta values
    learning_rate = K.backend.get_value(optimizer.learning_rate)
    beta_1 = K.backend.get_value(optimizer.beta_1)
    beta_2 = K.backend.get_value(optimizer.beta_2)
    
    print((learning_rate, beta_1, beta_2))
