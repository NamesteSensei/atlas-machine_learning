#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
learning_rate_decay = __import__('11-learning_rate_decay').learning_rate_decay

if __name__ == '__main__':
    alpha_init = 0.1
    decay_rate = 1
    decay_step = 10
    alphas = []

    for i in range(100):
        alpha = learning_rate_decay(alpha_init, decay_rate, i, decay_step)
        alphas.append(alpha)
        print(alpha)

    # Visualization
    plt.plot(range(100), alphas, marker='o', linestyle='-', color='b')
    plt.xlabel('Global Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Decay Over Time')
    plt.show()
