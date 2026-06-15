import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def batchnorm(x, gamma, beta, running_mean, running_var, training=True, momentum=0.9, eps=1e-5):
    mu = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    if training:
        x_hat = (x - mu) / np.sqrt(var + eps)
        running_mean[:] = momentum * running_mean + (1 - momentum) * mu
        running_var[:] = momentum * running_var + (1 - momentum) * var
    else:
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
    cache = (x_hat, gamma, beta, mu, var, eps)
    return gamma * x_hat + beta, cache

# print(softmax(np.array([1, 2, 3])))

d, n, k = 6, 32, 2
layer_sizes = [d, n, n, k]
L = len(layer_sizes) - 1
B = 128
weights, biases = [], []
for i in range(L):
    weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) / np.sqrt(layer_sizes[i]))
    biases.append(np.random.randn(layer_sizes[i+1],1) / np.sqrt(layer_sizes[i]))

# Initialize batch normalization parameters
gamma = [np.ones((layer_sizes[i+1], 1)) for i in range(L-1)]
beta = [np.zeros((layer_sizes[i+1], 1)) for i in range(L-1)]
running_means = [np.zeros((layer_sizes[i+1], 1)) for i in range(L-1)]
running_vars = [np.ones((layer_sizes[i+1], 1)) for i in range(L-1)]

def feedforward(W, x, b, gamma, beta, training=True):
    h = [x]
    a = []
    caches = []
    for i in range(L):
        a.append(W[i].T @ h[-1] + b[i])
        if i == L - 1:
            h.append(softmax(a[-1]))
        else:
            h_i, cache = batchnorm(a[-1], gamma[i], beta[i], running_means[i], running_vars[i], training)
            h.append(sigmoid(h_i))
            caches.append(cache)
    return h, a, caches

def backpropagation(W, h, y, gamma, beta, caches, eps=1e-5):
    batch_size = h[0].shape[1]
    grad_a = [h[-1] - y]
    grad_W = [(h[-2] @ grad_a[-1].T) / batch_size]
    grad_b = [np.mean(grad_a[-1], axis=1, keepdims=True)]
    grad_gamma = []
    grad_beta = []
    for i in range(L-2, -1, -1):
        G = (W[i+1] @ grad_a[-1]) * h[i+1] * (1 - h[i+1])  
        Q = G * gamma[i] 
        cache = caches[i]
        a_hat = cache[0]
        var = cache[4]
        grad_gamma.append(np.sum(G * a_hat, axis=1, keepdims=True))
        grad_beta.append(np.sum(G, axis=1, keepdims=True))
        grad_a.append((Q - np.mean(Q, axis=1, keepdims=True) - a_hat * np.mean(Q * a_hat, axis=1, keepdims=True)) / np.sqrt(var + eps))
        grad_W.append((h[i] @ grad_a[-1].T) / batch_size)
        grad_b.append(np.mean(grad_a[-1], axis=1, keepdims=True))
    grad_W.reverse()
    grad_a.reverse()
    grad_b.reverse()
    grad_gamma.reverse()
    grad_beta.reverse()
    return grad_W, grad_b, grad_gamma, grad_beta

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


eta = 0.01
epochs = 5000
mem = 0.9
v_W, v_b, v_gamma, v_beta = [np.zeros_like(W) for W in weights], [np.zeros_like(b) for b in biases], [np.zeros_like(g) for g in gamma], [np.zeros_like(b) for b in beta]

for epoch in range(epochs):
    perm = np.random.permutation(X.shape[0])
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]
    for start in range(0, X.shape[0], B):
        end = start + B
        lookahead_W = [W - mem * v for W, v in zip(weights, v_W)]
        lookahead_b = [b - mem * v for b, v in zip(biases, v_b)]
        lookahead_gamma = [g - mem * v for g, v in zip(gamma, v_gamma)]
        lookahead_beta = [b - mem * v for b, v in zip(beta, v_beta)]
        x_batch = X_shuffled[start:end].T
        y_batch = Y_shuffled[start:end].T
        h, a, caches = feedforward(lookahead_W, x_batch, lookahead_b, lookahead_gamma, lookahead_beta, training=True)
        
        grad_W, grad_b, grad_gamma, grad_beta = backpropagation(lookahead_W, h, y_batch, gamma, beta, caches)
        for i in range(L):
            v_W[i] = mem * v_W[i] + eta * grad_W[i]
            v_b[i] = mem * v_b[i] + eta * grad_b[i]
            weights[i] -= v_W[i]
            biases[i] -= v_b[i]
            if i < L-1:
                v_gamma[i] = mem * v_gamma[i] + eta * grad_gamma[i]
                v_beta[i] = mem * v_beta[i] + eta * grad_beta[i]
                gamma[i] -= eta * grad_gamma[i]
                beta[i] -= eta * grad_beta[i]

    # for x, y in zip(X, Y):
    #     x = x.reshape(-1,1)
    #     y = y.reshape(-1,1)
    #     lookahead_W = [W - mem * v for W, v in zip(weights, v_W)]
    #     lookahead_b = [b - mem * v for b, v in zip(biases, v_b)]
    #     h, a = feedforward(lookahead_W, x, lookahead_b)
    #     grad_W, grad_b = backpropagation(lookahead_W, h, y)
    #     # loss = -np.sum(y * np.log(h[-1]))
    #     # print(f"Loss: {loss}")
    #     for i in range(L):
    #         v_W[i] = mem * v_W[i] + eta * grad_W[i]
    #         v_b[i] = mem * v_b[i] + eta * grad_b[i]
    #         weights[i] -= v_W[i]
    #         biases[i] -= v_b[i]

# print("Final weights and biases:")
# for i in range(L):
#     print(f"Layer {i+1} weights:\n{weights[i]}")
#     print(f"Layer {i+1} biases:\n{biases[i]}")

# output = feedforward(weights, X[3], biases)[0][-1]
# print(f"Output for input [1, 1]:\n{output}")

# output = feedforward(weights, X[4].reshape(-1,1), biases, gamma, beta)[0][-1]
# print(f"Predicted output: {output}")
# real = Y[4].reshape(-1,1)
# print(f"Real label: {real}")

#accuracy
correct = 0
for x, y in zip(X, Y):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    output = feedforward(weights, x, biases, gamma, beta, training=False)[0][-1]
    if np.argmax(output) == np.argmax(y):
        correct += 1
print(f"Accuracy: {correct / len(X) * 100:.4f}%")
