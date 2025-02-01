# Neural Network for Binary Classification

## Project Overview
This project implements a **binary image classifier** using **NumPy** from scratch. The goal is to build a neural network that can recognize images based on a given dataset.

## Dataset
The project requires the following datasets, which should be stored in the `data/` directory:
- `Binary_Train.npz` - Training dataset.
- `Binary_Dev.npz` - Development dataset.
- `MNIST.npz` - MNIST handwritten digits dataset.

## Repository Structure
```
atlas-machine_learning/
│── supervised_learning/
│   ├── classification/
│   │   ├── 0-neuron.py
│   │   ├── 0-main.py
│   │   ├── 1-neuron.py
│   │   ├── ...
│── data/  
│   ├── Binary_Train.npz
│   ├── Binary_Dev.npz
│   ├── MNIST.npz
│── README.md
```

## Requirements
- Python 3.9
- NumPy 1.25.2
- Matplotlib (for visualization)

## Running the Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/atlas-machine_learning.git
   cd atlas-machine_learning
   ```
2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib
   ```
3. **Ensure data is placed in `data/` directory**
4. **Run test scripts:**
   ```bash
   python supervised_learning/classification/0-main.py
   ```

## Tasks Breakdown
- `0-neuron.py`: Implements a single neuron for binary classification.
  - Defines a class `Neuron` with initialization for weights, bias, and activation.
- `1-neuron.py`: Privatizes attributes in the neuron class.
- `2-neuron.py`: Implements forward propagation.
- `3-neuron.py`: Computes cost using logistic regression.
- `4-neuron.py`: Evaluates predictions.
- `5-neuron.py`: Implements gradient descent.
- `6-neuron.py`: Trains the model.
- `7-neuron.py`: Adds verbosity and visualization.
- `8-neural_network.py`: Implements a neural network with one hidden layer.
