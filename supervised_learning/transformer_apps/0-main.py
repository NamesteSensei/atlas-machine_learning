#!/usr/bin/env python3
"""
Main file for testing Dataset class from 0-dataset.py
"""

import importlib.util
import os

# Dynamically import Dataset from 0-dataset.py
module_name = 'dataset'
file_path = os.path.join(os.path.dirname(__file__), '0-dataset.py')

spec = importlib.util.spec_from_file_location(module_name, file_path)
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)

# Get the Dataset class from the loaded module
Dataset = dataset_module.Dataset

# Instantiate the Dataset and test it
data = Dataset()

for pt, en in data.data_train.take(2):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))

print(type(data.tokenizer_pt))
print(type(data.tokenizer_en))
