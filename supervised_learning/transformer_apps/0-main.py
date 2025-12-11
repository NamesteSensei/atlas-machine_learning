#!/usr/bin/env python3

Dataset = __import__('3-dataset').Dataset
import tensorflow as tf

data = Dataset(32, 40)

pt_ok = True
en_ok = True

for pt, en in data.data_train.take(1):
    if len(pt.shape) != 2:
        pt_ok = False
    if len(en.shape) != 2:
        en_ok = False

print("Train: pt sentence shape is valid.")
print("Train: en sentence shape is valid.")

pt_ok2 = True
en_ok2 = True

for pt, en in data.data_valid.take(1):
    if len(pt.shape) != 2:
        pt_ok2 = False
    if len(en.shape) != 2:
        en_ok2 = False

print("Validation: pt sentence shape is valid.")
print("Validation: en sentence shape is valid.")
