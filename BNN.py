import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def binarize(x):
    return np.where(x >=0, 1, -1)

# def batchnorm(x, gamma, beta, running_mean, running_var, momentum=0.9, training=True, eps=1e-5):
#     if training:
#         mu = np.mean(x, axis=1, keepdims=True)
#         var = np.var(x, axis=1, keepdims=True)
#         x_hat = (x - mu) / np.sqrt(var + eps)
#         # Update running statistics
#         running_mean = momentum * running_mean + (1 - momentum) * mu
#         running_var = momentum * running_var + (1 - momentum) * var
#     else:
#         x_hat = (x - running_mean) / np.sqrt(running_var + eps)
#     cache = (x_hat, gamma, beta, running_mean, running_var, eps)
#     return gamma * x_hat + beta, cache

# print(softmax(np.array([1, 2, 3])))

d, n, k = 6, 32, 2
layer_sizes = [d, n, n, n, k]
B = 32
L = len(layer_sizes) - 1
weights, biases, gamma, beta, running_means, running_vars = [], [], [], [], [], []
for i in range(L):
    weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
    biases.append(np.random.randn(layer_sizes[i+1],1))

# for i in range(L-1):
#     gamma.append(np.ones((layer_sizes[i+1], 1)))
#     beta.append(np.zeros((layer_sizes[i+1], 1)))
#     running_means.append(np.zeros((layer_sizes[i+1], 1)))
#     running_vars.append(np.ones((layer_sizes[i+1], 1)))

def feedforward(W, x, b):
    h = [x]
    a = []
    # bn_cache = []
    for i in range(L):
        Wb = binarize(W[i])
        a.append(Wb.T @ h[-1] + b[i])
        if i == L - 1:
            h.append(softmax(a[-1]))
        else:
            # z_norm, cache = batchnorm(a[-1], gamma[i], beta[i], running_means[i], running_vars[i], training=True)
            # bn_cache.append(cache)
            # h.append(sigmoid(z_norm))
            h.append(sigmoid(a[-1]))
    return h, a

# def batchnorm_backward(grad_a_bn, cache, eps=1e-5):
#     x_hat, gamma, beta, running_mean, running_var, eps = cache
#     grad_gamma = np.sum(grad_a_bn * x_hat, axis=1, keepdims=True)
#     grad_beta = np.sum(grad_a_bn, axis=1, keepdims=True)
#     grad_a = (gamma / np.sqrt(running_var + eps)) * (grad_a_bn - np.mean(grad_a_bn, axis=1, keepdims=True) - x_hat * np.mean(grad_a_bn * x_hat, axis=1, keepdims=True))
#     return grad_a, grad_gamma, grad_beta

def backpropagation(W, h, y, a):
    batch_size = h[0].shape[1]
    grad_a = [h[-1] - y]
    # grad_gamma = []
    # grad_beta = []
    grad_W = [(h[-2] @ grad_a[-1].T) / batch_size]
    grad_b = [np.sum(grad_a[-1], axis=1, keepdims=True) / batch_size]
    for i in range(L-2, -1, -1):
        Wb_next = binarize(W[i+1])
        # ste_mask = (np.abs(a[i]) <=1).astype(float)
        # grad_a_bn = (W[i+1] @ grad_a[-1]) * h[i+1] * (1 - h[i+1])
        # grad_a_current, grad_gamma_i, grad_beta_i = batchnorm_backward(grad_a_bn, bn_cache[i])
        # grad_gamma.append(grad_gamma_i)
        # grad_beta.append(grad_beta_i)
        mask = (np.abs(W[i]) <=1).astype(float)  
        grad_a.append((Wb_next @ grad_a[-1]) * h[i+1] * (1 - h[i+1]))
        grad_W.append((h[i] @ grad_a[-1].T) / batch_size)
        grad_b.append(np.sum(grad_a[-1], axis=1, keepdims=True) / batch_size)
        grad_W[-1] *= mask
    grad_W.reverse()
    grad_b.reverse()
    # grad_gamma.reverse()
    # grad_beta.reverse()
    return grad_W, grad_b  #, grad_gamma, grad_beta

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


eta = 1
beta = 0.9
epochs = 5000
v_W, v_b = [np.zeros_like(w) for w in weights], [np.zeros_like(b) for b in biases]

for epoch in range(epochs):
    perm = np.random.permutation(X.shape[0])
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]
    for start in range(0, X.shape[0], B):
        end = start + B
        lookahead_W = [w - beta * v for w, v in zip(weights, v_W)]
        lookahead_b = [b - beta * v for b, v in zip(biases, v_b)]
        x_batch = X_shuffled[start:end].T
        y_batch = Y_shuffled[start:end].T
        h, a = feedforward(lookahead_W, x_batch, lookahead_b)
        grad_W, grad_b = backpropagation(lookahead_W, h, y_batch, a)
        for i in range(L):
            v_W[i] = beta * v_W[i] + (1 - beta) * grad_W[i]
            v_b[i] = beta * v_b[i] + (1 - beta) * grad_b[i]
            weights[i] -= eta * v_W[i]
            biases[i] -= eta * v_b[i]
            weights[i] = np.clip(weights[i], -1, 1)

    # for x, y in zip(X, Y):
    #     x = x.reshape(-1,1)
    #     y = y.reshape(-1,1)
    #     h, a = feedforward(weights, x, biases, gamma, beta)
    #     grad_W, grad_b = backpropagation(weights, h, y)
    #     # loss = -np.sum(y * np.log(h[-1]))
    #     # print(f"Loss: {loss}")
    #     for i in range(L):
    #         weights[i] -= eta * grad_W[i]
    #         biases[i] -= eta * grad_b[i]

# print("Final weights and biases:")
# for i in range(L):
#     print(f"Layer {i+1} weights:\n{weights[i]}")
#     print(f"Layer {i+1} biases:\n{biases[i]}")

# output = feedforward(weights, X[3], biases)[0][-1]
# print(f"Output for input [1, 1]:\n{output}")

# output = feedforward(weights, X[51].reshape(-1,1), biases)[0][-1]
# print(f"Predicted output: {output}")
# real = Y[51].reshape(-1,1)
# print(f"Real label: {real}")

#accuracy
correct = 0
for x, y in zip(X, Y):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    output = feedforward(weights, x, biases)[0][-1]
    if np.argmax(output) == np.argmax(y):
        correct += 1

print(f"Accuracy: {correct / len(X) * 100:.4f}%")