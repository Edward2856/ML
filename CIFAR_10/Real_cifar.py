import numpy as np
import torch
from torchvision import datasets, transforms
# import os
# print(os.getcwd())

# torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

def sigmoid(z):
    return torch.sigmoid(z)

def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    exp_z = torch.exp(z - torch.max(z, dim=0, keepdim=True)[0])  # for numerical stability
    return exp_z / torch.sum(exp_z, dim=0, keepdim=True)

def batchnorm(x, gamma, beta, running_mean, running_var, training=True, momentum=0.9, eps=1e-5):
    mu = torch.mean(x, dim=1, keepdim=True)
    var = torch.var(x, dim=1, keepdim=True, unbiased=False)
    if training:
        x_hat = (x - mu) / torch.sqrt(var + eps)
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
    else:
        x_hat = (x - running_mean) / torch.sqrt(running_var + eps)
    cache = (x_hat, gamma, beta, mu, var, eps)
    return gamma * x_hat + beta, cache, running_mean, running_var

d, n, k = 32*32*3, 1024, 10
layer_sizes = [d, n, n, n, k]
L = len(layer_sizes) - 1
B = 1500
weights, biases = [], []
for i in range(L):
    weights.append(torch.randn(layer_sizes[i], layer_sizes[i+1], device=device) / np.sqrt(layer_sizes[i]))
    biases.append(torch.randn(layer_sizes[i+1], 1, device=device) / np.sqrt(layer_sizes[i]))

# Initialize batch normalization parameters
gamma = [torch.ones((layer_sizes[i+1], 1), device=device) for i in range(L-1)]
beta = [torch.zeros((layer_sizes[i+1], 1), device=device) for i in range(L-1)]
running_means = [torch.zeros((layer_sizes[i+1], 1), device=device) for i in range(L-1)]
running_vars = [torch.ones((layer_sizes[i+1], 1), device=device) for i in range(L-1)]

def feedforward(W, x, b, gamma, beta, training=True):
    h = [x]
    a = []
    caches = []
    for i in range(L):
        a.append(W[i].T @ h[-1] + b[i])
        if i == L - 1:
            h.append(softmax(a[-1]))
        else:
            h_i, cache, running_means[i], running_vars[i] = batchnorm(a[-1], gamma[i], beta[i], running_means[i], running_vars[i], training)
            h.append(sigmoid(h_i))
            caches.append(cache)
    return h, a, caches

def backpropagation(W, h, y, gamma, beta, caches, eps=1e-5):
    batch_size = h[0].shape[1]
    grad_a = [h[-1] - y]
    grad_W = [(h[-2] @ grad_a[-1].T) / batch_size]
    grad_b = [torch.mean(grad_a[-1], dim=1, keepdim=True)]
    grad_gamma = []
    grad_beta = []
    for i in range(L-2, -1, -1):
        G = (W[i+1] @ grad_a[-1]) * h[i+1] * (1 - h[i+1])  
        Q = G * gamma[i] 
        cache = caches[i]
        a_hat = cache[0]
        var = cache[4]
        grad_gamma.append(torch.sum(G * a_hat, dim=1, keepdim=True))
        grad_beta.append(torch.sum(G, dim=1, keepdim=True))
        grad_a.append((Q - torch.mean(Q, dim=1, keepdim=True) - a_hat * torch.mean(Q * a_hat, dim=1, keepdim=True)) / torch.sqrt(var + eps))
        grad_W.append((h[i] @ grad_a[-1].T) / batch_size)
        grad_b.append(torch.mean(grad_a[-1], dim=1, keepdim=True))
    grad_W.reverse()
    grad_a.reverse()
    grad_b.reverse()
    grad_gamma.reverse()
    grad_beta.reverse()
    return grad_W, grad_b, grad_gamma, grad_beta

# CIFAR-10 Dataset

# train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# X = torch.stack([train_dataset[i][0].view(-1) for i in range(len(train_dataset))], dim=1).to(device)
# labels = torch.tensor(train_dataset.targets, device=device)
# Y = torch.nn.functional.one_hot(labels, num_classes=10).float().T

# X_t = torch.stack([test_dataset[i][0].view(-1) for i in range(len(test_dataset))], dim=1).to(device)
# test_labels = torch.tensor(test_dataset.targets, device=device)
# Y_t = torch.nn.functional.one_hot(test_labels, num_classes=10).float().T

# torch.save(X, 'train_images.pt')
# torch.save(Y, 'train_labels.pt')
# torch.save(X_t, 'test_images.pt')
# torch.save(Y_t, 'test_labels.pt')

X = torch.load('train_images.pt', map_location=device)
Y = torch.load('train_labels.pt', map_location=device)

X_t = torch.load('test_images.pt', map_location=device)
Y_t = torch.load('test_labels.pt', map_location=device)

eta = 0.7
epochs = 1000
mem = 0.8
v_W, v_b, v_gamma, v_beta = [torch.zeros_like(W, device=device) for W in weights], [torch.zeros_like(b, device=device) for b in biases], [torch.zeros_like(g, device=device) for g in gamma], [torch.zeros_like(b, device=device) for b in beta]
accuracy_history = []

for epoch in range(epochs):
    perm = torch.randperm(X.shape[1], device=device)
    for start in range(0, X.shape[1], B):
        end = start + B
        idx = perm[start:end]
        lookahead_W = [W - mem * v for W, v in zip(weights, v_W)]
        lookahead_b = [b - mem * v for b, v in zip(biases, v_b)]
        lookahead_gamma = [g - mem * v for g, v in zip(gamma, v_gamma)]
        lookahead_beta = [b - mem * v for b, v in zip(beta, v_beta)]
        x_batch = X[:,idx]
        y_batch = Y[:,idx]
        h, a, caches = feedforward(lookahead_W, x_batch, lookahead_b, lookahead_gamma, lookahead_beta, training=True)
        
        grad_W, grad_b, grad_gamma, grad_beta = backpropagation(lookahead_W, h, y_batch, lookahead_gamma, lookahead_beta, caches)
        for i in range(L):
            v_W[i].mul_(mem).add_(grad_W[i], alpha=eta)
            v_b[i].mul_(mem).add_(grad_b[i], alpha=eta)
            weights[i].sub_(v_W[i])
            biases[i].sub_(v_b[i])
            # v_W[i] = mem * v_W[i] + eta * grad_W[i]
            # v_b[i] = mem * v_b[i] + eta * grad_b[i]
            # weights[i] -= v_W[i]
            # biases[i] -= v_b[i]
            if i < L-1:
                v_gamma[i].mul_(mem).add_(grad_gamma[i], alpha=eta)
                v_beta[i].mul_(mem).add_(grad_beta[i], alpha=eta)
                gamma[i].sub_(v_gamma[i])
                beta[i].sub_(v_beta[i])
                # v_gamma[i] = mem * v_gamma[i] + eta * grad_gamma[i]
                # v_beta[i] = mem * v_beta[i] + eta * grad_beta[i]
                # gamma[i] -= v_gamma[i]
                # beta[i] -= v_beta[i]

    with torch.no_grad():
        output = feedforward(weights, X, biases, gamma, beta, training=False)[0][-1]
        pred = torch.argmax(output, dim=0)
        true = torch.argmax(Y, dim=0)
        accuracy_train = (pred == true).float().mean().item()

        output = feedforward(weights, X_t, biases, gamma, beta, training=False)[0][-1]
        pred = torch.argmax(output, dim=0)
        true = torch.argmax(Y_t, dim=0)
        accuracy_test = (pred == true).float().mean().item()
        accuracy_history.append(accuracy_test)

        print(
            f'Epochs:{epoch+1}/{epochs} | '
            f'Testing Accuracy: {accuracy_test * 100:.2f} | '
            f'Training Accuracy: {accuracy_train * 100:.2f}'
        )

np.save("Real_weighted_model_accuracy.npy", np.array(accuracy_history))