#!/usr/bin/env python3
import numpy as np
pool = __import__('6-pool').pool

if __name__ == '__main__':
    dataset = np.load('animals_1.npz')
    images = dataset['data']
    print(images.shape)  # (10000, 32, 32, 3)

    images_pool = pool(images, (2, 2), (2, 2), mode='avg')
    print(images_pool.shape)  # Expect: (10000, 16, 16, 3)

    # Print a slice of pooled image for autograder check
    print(images_pool[0, :5, :5, 0])
