# Hidden Markov Models (HMM) — Machine Learning Project

## 📘 Overview

This project explores **Markov Chains** and **Hidden Markov Models (HMMs)** — powerful mathematical frameworks for modeling systems that evolve probabilistically over time.  
You’ll learn how to **simulate, analyze, and decode** Markov processes using pure Python and NumPy.

By the end, you will understand:
- How states transition probabilistically (Markov Chains)
- How hidden states generate observable outputs (HMMs)
- How algorithms like **Forward**, **Backward**, **Viterbi**, and **Baum-Welch** reveal or train these models

---

## 🧩 Concepts Covered

| Concept | Description |
|----------|-------------|
| **Markov Property** | The future depends only on the present, not the past. |
| **Transition Matrix** | Describes the probability of moving between states. |
| **Stationary Distribution** | Long-run equilibrium probabilities for each state. |
| **Regular Markov Chains** | Chains that eventually reach a steady state regardless of start. |
| **Absorbing States** | Once entered, cannot be left (probability = 1). |
| **Hidden Markov Model (HMM)** | System with hidden states that emit observable outputs. |
| **Emission Matrix** | Defines the probability of each observation given a hidden state. |
| **Forward Algorithm** | Calculates the likelihood of an observation sequence. |
| **Backward Algorithm** | Computes probabilities from the end backwards. |
| **Viterbi Algorithm** | Finds the most probable sequence of hidden states. |
| **Baum-Welch Algorithm** | Learns model parameters using Expectation-Maximization. |

---

## ⚙️ Project Structure


---

## 🧠 What You’ll Learn

1. **Probability Transitions** — How systems evolve step by step  
2. **Steady States & Convergence** — How probabilities stabilize over time  
3. **Hidden Models** — How unseen causes lead to observed effects  
4. **Dynamic Programming in HMMs** — Efficient algorithms to compute, decode, and learn  
5. **Numerical Stability** — Why log-sum-exp and scaling matter in real models  

---

## 💻 Requirements

- Ubuntu 20.04 LTS  
- Python 3.9  
- NumPy 1.25.2  
- Only standard libraries + `import numpy as np` allowed  
- All files executable  
- All modules, classes, and functions must be documented  
- PEP8 (pycodestyle 2.11.1) compliant  

---

## 🧪 Running Examples

Example for Task 0:
```bash
$ ./0-main.py
[[0.2494929  0.26335362 0.23394185 0.25321163]]

