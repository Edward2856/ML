import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# print(softmax(np.array([1, 2, 3])))

d, n, k = 6, 10, 2
layer_sizes = [d, n, n, k]
L = len(layer_sizes) - 1
B = 32
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

def backpropagation(W, h, y):
    batch_size = h[0].shape[1]
    grad_a = [h[-1] - y]
    grad_W = [(h[-2] @ grad_a[-1].T) / batch_size]
    grad_b = [np.mean(grad_a[-1], axis=1, keepdims=True)]
    for i in range(L-2, -1, -1):
        grad_a.append((W[i+1] @ grad_a[-1]) * h[i+1] * (1 - h[i+1]))
        grad_W.append((h[i] @ grad_a[-1].T) / batch_size)
        grad_b.append(np.mean(grad_a[-1], axis=1, keepdims=True))
    grad_W.reverse()
    grad_a.reverse()
    grad_b.reverse()
    return grad_W, grad_b

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

# # XOR gate training data
# X = [
#     np.array([[0],[0]]),
#     np.array([[0],[1]]),
#     np.array([[1],[0]]),
#     np.array([[1],[1]])
# ]

# Y = [
#     np.array([[1],[0]]),
#     np.array([[0],[1]]),
#     np.array([[0],[1]]),
#     np.array([[1],[0]])
# ]

# Titanic dataset
data = pd.read_csv(r"D:\NITPY\IITM Internship\ML\train.csv")
data["Age"] = data["Age"].fillna(data["Age"].median())
X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].to_numpy()
X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)  # Normalize features
Y = data["Survived"].to_numpy().reshape(-1, 1)
Y = np.hstack((1 - Y, Y))  # Convert to one-hot encoding


eta = 0.5
epochs = 1000
gamma = 0.9
v_W, v_b = [np.zeros_like(W) for W in weights], [np.zeros_like(b) for b in biases]

for epoch in range(epochs):
    perm = np.random.permutation(X.shape[0])
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]
    for start in range(0, X.shape[0], B):
        end = start + B
        lookahead_W = [W - gamma * v for W, v in zip(weights, v_W)]
        lookahead_b = [b - gamma * v for b, v in zip(biases, v_b)]
        x_batch = X_shuffled[start:end].T
        y_batch = Y_shuffled[start:end].T
        h, a = feedforward(lookahead_W, x_batch, lookahead_b)
        grad_W, grad_b = backpropagation(lookahead_W, h, y_batch)
        for i in range(L):
            v_W[i] = gamma * v_W[i] + eta * grad_W[i]
            v_b[i] = gamma * v_b[i] + eta * grad_b[i]
            weights[i] -= v_W[i]
            biases[i] -= v_b[i]
    # for x, y in zip(X, Y):
    #     x = x.reshape(-1,1)
    #     y = y.reshape(-1,1)
    #     lookahead_W = [W - gamma * v for W, v in zip(weights, v_W)]
    #     lookahead_b = [b - gamma * v for b, v in zip(biases, v_b)]
    #     h, a = feedforward(lookahead_W, x, lookahead_b)
    #     grad_W, grad_b = backpropagation(lookahead_W, h, y)
    #     # loss = -np.sum(y * np.log(h[-1]))
    #     # print(f"Loss: {loss}")
    #     for i in range(L):
    #         v_W[i] = gamma * v_W[i] + eta * grad_W[i]
    #         v_b[i] = gamma * v_b[i] + eta * grad_b[i]
    #         weights[i] -= v_W[i]
    #         biases[i] -= v_b[i]

# print("Final weights and biases:")
# for i in range(L):
#     print(f"Layer {i+1} weights:\n{weights[i]}")
#     print(f"Layer {i+1} biases:\n{biases[i]}")

# output = feedforward(weights, X[3], biases)[0][-1]
# print(f"Output for input [1, 1]:\n{output}")

output = feedforward(weights, X[51].reshape(-1,1), biases)[0][-1]
print(f"Predicted output: {output}")
real = Y[51].reshape(-1,1)
print(f"Real label: {real}")
