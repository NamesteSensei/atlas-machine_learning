# Transformer Applications – Dataset Loader

This project implements the dataset loading and tokenizer preparation required for machine translation using the TED HRLR Portuguese-to-English dataset.

## Files

### 0-dataset.py
Defines the `Dataset` class responsible for:

- Loading the training and validation splits from `ted_hrlr_translate/pt_to_en`
- Creating pretrained tokenizers:
  - Portuguese tokenizer: `neuralmind/bert-base-portuguese-cased`
  - English tokenizer: `bert-base-uncased`

The class contains:
- `data_train`: training dataset
- `data_valid`: validation dataset
- `tokenizer_pt`: Portuguese tokenizer
- `tokenizer_en`: English tokenizer

All components follow the required specifications, including documentation and pycodestyle formatting.

### 0-main.py
This script tests the `Dataset` class by:

- Printing one example from the training split
- Printing one example from the validation split
- Displaying the types of the created tokenizers

## Requirements Satisfied

- Correct dataset loading from TensorFlow Datasets
- Valid pretrained tokenizers using the transformers library
- Only allowed imports used
- Each file starts with `#!/usr/bin/env python3`
- All code documented
- All files pycodestyle compliant

## Usage

Run the test file:

```bash
./0-main.py
