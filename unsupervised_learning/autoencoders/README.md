# Autoencoders - Unsupervised Learning Project

This project explores **autoencoders**, a type of neural network used for **unsupervised learning**. The goal is to learn compact, meaningful representations of data by forcing the network to reconstruct its input through a bottleneck.

## 📚 What You'll Learn

- What is an autoencoder?
- What is latent space?
- What is a bottleneck?
- How sparse autoencoders differ from vanilla ones
- How convolutional autoencoders handle images
- What is a variational autoencoder (VAE)?
- What is Kullback-Leibler divergence (KL divergence)?

## 🧠 Core Idea

An autoencoder is like a photocopier that compresses and recreates your data. It first **encodes** the input to a lower-dimensional form, then **decodes** it back. If trained correctly, this bottleneck forces the model to learn what's truly essential in the data.

## 🛠️ Technologies Used

- Python 3.9
- TensorFlow 2.15
- NumPy 1.25.2
- Matplotlib
- Pycodestyle 2.11.1

## 🗂️ Project Structure

```bash
autoencoders/
├── 0-vanilla.py            # Basic fully connected autoencoder
├── 1-sparse.py             # Sparse autoencoder with L1 regularization
├── 2-convolutional.py      # Convolutional autoencoder for image data
├── 3-variational.py        # Variational autoencoder with KL divergence
├── README.md               # This file
