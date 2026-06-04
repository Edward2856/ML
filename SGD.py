import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # for numerical stability
    return exp_z / np.sum(exp_z)

# print(softmax(np.array([1, 2, 3])))

d, n, k = 2, 4, 2
layer_sizes = [d, n, k]
L = len(layer_sizes) - 1
weights, biases = [], []
for i in range(L):
    weights.append(np.random.rand(layer_sizes[i], layer_sizes[i+1]))
    biases.append(np.random.rand(layer_sizes[i+1],1))

def feedforward(W, x, b):
    h = [x]
    a = []
    for i in range(L):
        a.append(W[i].T @ h[-1] + b[i])
        if i == L - 1:
            h.append(softmax(a[-1]))
        else:
            h.append(sigmoid(a[-1]))
    return h, a

def backpropagation(W, h, a, y):
    grad_a = [h[-1] - y]
    grad_W = [h[-2] @ grad_a[-1].T]
    # grad_b = [grad_a[-1]]
    for i in range(L-2, -1, -1):
        grad_a.append((W[i+1] @ grad_a[-1]) * h[i+1] * (1 - h[i+1]))
        grad_W.append(h[i] @ grad_a[-1].T)
        # grad_b.append(grad_a[-1])
    grad_W.reverse()
    grad_a.reverse()
    return grad_W, grad_a

# #AND gate training data
# X = [
#     np.array([[0],[0]]),
#     np.array([[0],[1]]),
#     np.array([[1],[0]]),
#     np.array([[1],[1]])
# ]

# Y = [
#     np.array([[1],[0]]),
#     np.array([[1],[0]]),
#     np.array([[1],[0]]),
#     np.array([[0],[1]])
# ]

# XOR gate training data
X = [
    np.array([[0],[0]]),
    np.array([[0],[1]]),
    np.array([[1],[0]]),
    np.array([[1],[1]])
]

Y = [
    np.array([[1],[0]]),
    np.array([[0],[1]]),
    np.array([[0],[1]]),
    np.array([[1],[0]])
]

# # Titanic dataset
# data = pd.read_csv(r"D:\NITPY\IITM Internship\ML\train.csv")
# data["Age"] = data["Age"].fillna(data["Age"].median())
# X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].to_numpy()
# Y = data["Survived"].to_numpy().reshape(-1, 1)
# Y = np.hstack((1 - Y, Y))  # Convert to one-hot encoding


eta = 10
epochs = 1000
for epoch in range(epochs):
    for x, y in zip(X, Y):
        # x = x.reshape(-1,1)
        # y = y.reshape(-1,1)
        h, a = feedforward(weights, x, biases)
        grad_W, grad_b = backpropagation(weights, h, a, y)
        # loss = -np.sum(y * np.log(h[-1]))
        # print(f"Loss: {loss}")
        for i in range(L):
            weights[i] -= eta * grad_W[i]
            biases[i] -= eta * grad_b[i]

# print("Final weights and biases:")
# for i in range(L):
#     print(f"Layer {i+1} weights:\n{weights[i]}")
#     print(f"Layer {i+1} biases:\n{biases[i]}")

output = feedforward(weights, X[3], biases)[0][-1]
print(f"Output for input [1, 1]:\n{output}")

# output = feedforward(weights, X[1].reshape(-1,1), biases)[0][-1]
# print(f"Predicted output: {output}")
# real = Y[1].reshape(-1,1)
# print(f"Real label: {real}")
