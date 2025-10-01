#!/usr/bin/env python3

import numpy as np
expectation = __import__('6-expectation').expectation
initialize = __import__('4-initialize').initialize

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)

    pi, m, S = initialize(X, 4)
    g, likelihood = expectation(X, pi, m, S)

    print(g)
    print(likelihood)
