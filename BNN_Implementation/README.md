# Binary Neural Network (BNN) Implementation

A PyTorch implementation of a **Binary Neural Network (BNN)** based on the BinaryNet architecture, developed by Hubara et al.

The primary objective of this project was not only to reproduce the results of the original paper, but also to gain a deep understanding of Binary Neural Networks by first implementing conventional neural networks from scratch before transitioning to PyTorch.

---

## Project Overview

Binary Neural Networks replace floating-point weights (and optionally activations) with binary values (`+1` and `-1`), significantly reducing memory usage and computational complexity while maintaining competitive classification performance.

This repository contains:

* Fully connected Binary Neural Networks
* Convolutional Binary Neural Networks
* Weight binarization using the Straight-Through Estimator (STE)
* Batch Normalization
* Training and evaluation scripts
* Support for multiple benchmark datasets

---

## Features

* Binary weight implementation using the Straight-Through Estimator
* Batch Normalization
* Learning rate scheduling
* Data augmentation support
* GPU acceleration using PyTorch
* Training and testing scripts
* Accuracy logging
* Reproduction of BinaryNet experiments

---

## Datasets

The implementation has been evaluated on:

* MNIST
* CIFAR-10
* SVHN

Dataset preprocessing and augmentation have been done as opposed to the original paper.

---

## Repository Structure

```text
.
├── network.py               # BNN architecture like necessary functions
├── train.py                 # Training script
├── mnist.py                 # MNIST architecture
├── cifar.py                 # CIFAR-10 architecture
├── svhn.py                  # SVHN architecture
├── network_real.py          # Real-weighted model
├── train_real.py            # Training script for the real model
├── results/                 # Accuracy plots and logs
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Edward2856/ML/BNN_Implementation.git
```

Install the required packages

```bash
pip install torch torchvision numpy matplotlib
```

---

## Training

Example:

```bash
python train.py
```

Depending on the configuration, the script supports training on MNIST, CIFAR-10, or SVHN.

---

## Results

After reproducing the complete training pipeline—including preprocessing, hyperparameter settings, and data augmentation—the implementation achieves performance comparable to the results reported in the original BinaryNet paper.

| Dataset  | Reproduced Performance |
| -------- | ---------------------- |
| MNIST    | Comparable to paper    |
| CIFAR-10 | Comparable to paper    |
| SVHN     | Comparable to paper    |

---

## Implementation Details

The implementation includes:

* Binary weight quantization
* Straight-Through Estimator (STE) for gradient propagation
* Batch Normalization
* Mini-batch Gradient Descent
* Nesterov Momentum
* Learning rate decay
* Data augmentation for image datasets

The goal was to remain as faithful as possible to the methodology described in the original publication while adapting the implementation to modern PyTorch APIs.

---

## Learning Outcomes

This project involved:

* Understanding neural networks from first principles
* Implementing forward and backward propagation manually (prior NumPy implementation)
* Transitioning to PyTorch for efficient GPU-based experimentation
* Reproducing a published research paper
* Hyperparameter tuning and debugging
* Performance evaluation on multiple benchmark datasets

---

## Future Improvements

* Additional Binary Neural Network architectures
* ImageNet support
* Mixed-precision experiments
* Hardware-aware optimization
* ONNX export
* Model quantization comparisons

---

## References

Courbariaux, M., Hubara, I., Soudry, D., El-Yaniv, R., & Bengio, Y.

**BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.**

https://arxiv.org/abs/1602.02830

---

## Author

Edward Saveri
