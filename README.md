# Deep Learning from Scratch

This repository documents my journey of learning and implementing Machine Learning and Deep Learning algorithms from first principles as part of my Summer Research Internship (May–July 2026).

Unlike the accompanying BNN_Implementation repository, this repository focuses on understanding the underlying mathematics and implementation details of neural networks rather than reproducing a single research paper.

It contains hand-coded implementations, experimental code, intermediate versions, and rough work developed while studying the foundations of modern deep learning.

---

## Objectives

The primary goals of this repository were to:

* Understand the mathematical foundations of neural networks.
* Implement every stage of training manually before relying on deep learning frameworks.
* Experiment with different optimization techniques.
* Build intuition for gradient-based learning.
* Develop the knowledge required to reproduce a Binary Neural Network research paper.

---

## Topics Covered

The repository includes implementations and experiments involving:

* Single Artificial Neuron
* Feedforward Neural Networks
* Forward Propagation
* Backpropagation
* Gradient Descent
* Mini-batch Gradient Descent
* Stochastic Gradient Descent
* Momentum
* Nesterov Accelerated Gradient (NAG)
* Activation Functions
* Loss Functions
* Batch Normalization
* Binary Weight Binarization
* Straight-Through Estimator (STE)
* Convolutional Neural Networks (CNNs)
* PyTorch implementations
* NumPy implementations

---

## Datasets Used

Various datasets were used during different stages of development.

* Titanic (Kaggle)
* Breast Cancer Wisconsin Dataset
* MNIST
* CIFAR-10
* SVHN

Smaller datasets were primarily used to validate mathematical correctness before scaling to larger image classification benchmarks.

---

## Repository Structure

The repository consists of several independent experiments rather than a single software project.

Examples include:

```text
MATLAB/
    Single Neuron

Python/
    Forward Propagation
    Backpropagation
    Feedforward Networks
    Batch Normalization
    Optimization Algorithms
    Binary Neural Networks
    CNN Experiments
    PyTorch Versions
```

Many files represent intermediate implementations developed while learning or debugging a particular concept.

---

## Development Journey

The work in this repository roughly followed the progression below:

1. Learning the fundamentals of Machine Learning and Deep Learning.
2. Implementing a single neuron in MATLAB.
3. Building fully connected neural networks using NumPy.
4. Implementing forward and backward propagation manually.
5. Adding optimization algorithms.
6. Implementing Batch Normalization from scratch.
7. Exploring Binary Neural Networks.
8. Transitioning to PyTorch for larger-scale experiments.
9. Developing convolutional neural networks.
10. Reproducing the BinaryNet research paper (available in the separate repository).

---

## Notes

This repository is intended primarily as a learning archive.

The code was written progressively while exploring different concepts, so coding style and project structure may vary between folders. Earlier implementations prioritize clarity and understanding over efficiency, whereas later implementations incorporate PyTorch, GPU acceleration, and more scalable training pipelines.

Several programs are experimental in nature and were created to verify individual concepts before integrating them into larger projects.

---

## Related Repository

The final outcome of this work is the **BNN_Implementation** repository, which contains a cleaner implementation aimed at reproducing the BinaryNet research paper and benchmarking it on MNIST, CIFAR-10, and SVHN.

---

## Skills Developed

Through the work in this repository, I gained experience in:

* Neural network implementation from scratch
* Matrix calculus for backpropagation
* Numerical optimization
* Batch Normalization
* Binary Neural Networks
* CNNs
* PyTorch
* NumPy
* Hyperparameter tuning
* Research paper implementation
* Machine learning experimentation

---

## Author

Edward Saveri
