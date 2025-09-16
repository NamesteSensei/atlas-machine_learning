# Bayesian Probability - Drug Side Effect Analysis

This project implements core Bayesian probability concepts using Python and NumPy. The context is a medical trial where we estimate the probability of patients developing severe side effects after taking a new cancer drug.

## 📘 Concepts Covered

- **Likelihood**: Probability of observed data given a hypothesis.
- **Intersection**: Joint probability of both data and hypothesis.
- **Marginal**: Total probability of the data across all hypotheses.
- **Posterior**: Updated belief about the hypothesis after seeing data.

## 🛠️ Files

| File              | Description |
|-------------------|-------------|
| `0-likelihood.py` | Computes likelihood using binomial distribution |
| `1-intersection.py` | Multiplies likelihood and prior (joint probability) |
| `2-marginal.py`   | Sums over all joint probabilities |
| `3-posterior.py`  | Applies Bayes’ theorem to get posterior probabilities |

## 🧪 Usage

Each task includes a `*-main.py` test file. Run them with:

```bash
chmod +x *.py
./0-main.py
./1-main.py
...

