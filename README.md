# Neural-Network-from-scratch
neural-network-from-scratch/
│
├── README.md
├── requirements.txt
├── neural_network.py
├── train.py
├── predict.py
├── utils.py
├── notebooks/
│   └── neural_network_demo.ipynb
├── results/
│   └── training_curve.png
└── .gitignore
# Neural Network From Scratch

A complete educational implementation of a feed-forward neural network using **Python + NumPy**, without TensorFlow or PyTorch for the core neural-network logic.

## What this project demonstrates

- Forward propagation
- ReLU and Softmax activations
- Cross-entropy loss
- Backpropagation
- Gradient descent
- Mini-batch training
- Xavier/He-style parameter initialization
- MNIST digit classification
- Training-loss and accuracy visualization
- Model save/load with NumPy

## Architecture

```text
MNIST image
    ↓
784 input features
    ↓
Dense layer: 128
    ↓
ReLU
    ↓
Dense layer: 64
    ↓
ReLU
    ↓
Dense layer: 10
    ↓
Softmax
    ↓
Predicted digit 0–9


## Installation

```bash
git clone https://github.com/YOUR_USERNAME/neural-network-from-scratch.git
cd neural-network-from-scratch
pip install -r requirements.txt
```

## Run MNIST training

```bash
python src/train_mnist.py
```

The script downloads MNIST through `keras.datasets` only as a **data-loading utility**. TensorFlow/Keras is not used to build, train, or predict with the neural network.

## Run prediction

After training:

```bash
python src/predict.py
```

## Core learning

The network computes:

### Forward propagation

`Z = XW + b`

`A = ReLU(Z)`

For the output:

`P = softmax(Z)`

### Softmax

`softmax(z_i) = exp(z_i) / Σ exp(z_j)`

### Cross-entropy

`L = -Σ y_i log(p_i)`

### Gradient descent

`W = W - learning_rate × dW`

### Backpropagation

Gradients are propagated from the output layer toward the input so that the weights can be updated to reduce the loss.

## Interview explanation

> I built a feed-forward neural network from scratch using NumPy. I implemented dense layers, ReLU, softmax, cross-entropy loss, forward propagation, backpropagation, mini-batch gradient descent, and parameter initialization manually. I tested the implementation first on XOR and then extended it to MNIST digit classification.

## Important distinction

This project intentionally does **not** use PyTorch or TensorFlow for the neural-network implementation. The purpose is to understand what frameworks automate internally.

## Suggested next upgrades

1. Add dropout.
2. Add L2 regularization.
3. Add configurable network depth.
4. Add Adam optimizer.
5. Add gradient checking.
6. Compare the implementation against PyTorch.
7. Build a CNN from scratch.
8. Apply the same learning principles to a healthcare dataset
```
