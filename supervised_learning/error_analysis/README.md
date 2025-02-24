# Machine Learning Error Analysis

## Overview

This project focuses on understanding and implementing error analysis in machine learning. We will create functions to evaluate model performance using metrics such as confusion matrices, sensitivity, specificity, precision, and F1 score.

## Topics Covered

- Confusion Matrix
- Type I & Type II Errors
- Sensitivity (Recall)
- Specificity
- Precision
- F1 Score
- Bias-Variance Tradeoff
- Bayes Error Rate

## Requirements

- Python 3.9
- NumPy 1.25.2
- Code should follow `pycodestyle` (2.11.1)
- All scripts must be executable
- Use `#!/usr/bin/env python3` as the first line in all scripts
- A README.md file is required at the root of the project

## Files

### Python Scripts:

1. `0-create_confusion.py` - Creates a confusion matrix
2. `1-sensitivity.py` - Calculates sensitivity for each class
3. `2-precision.py` - Calculates precision for each class
4. `3-specificity.py` - Calculates specificity for each class
5. `4-f1_score.py` - Calculates F1 score for each class

### Other Files:

- `5-error_handling` - Explains how to handle different bias-variance scenarios
- `6-compare_and_contrast` - Identifies key issues in model performance

## How to Run

Each Python script can be executed individually. Example:

```sh
./0-create_confusion.py
```

Ensure `labels_logits.npz` is available for confusion matrix calculations.
