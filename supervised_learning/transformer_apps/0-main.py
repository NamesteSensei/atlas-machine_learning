#!/usr/bin/env python3

Dataset = __import__('0-dataset').Dataset

data = Dataset()

# Print one sample from training
for pt, en in data.data_train.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))

# Print one sample from validation
for pt, en in data.data_valid.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))

# Print tokenizer types
print(type(data.tokenizer_pt))
print(type(data.tokenizer_en))

