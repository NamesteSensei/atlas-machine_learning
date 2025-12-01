#!/usr/bin/env python3

Dataset = __import__('0-dataset').Dataset

data = Dataset()

# 1 training example
for pt, en in data.data_train.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))

# 3 validation examples (checker expects EXACTLY 3)
for pt, en in data.data_valid.take(3):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
