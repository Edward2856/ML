import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Device name: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

def sigmoid(z):
    return torch.sigmoid(z)

def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    exp_z = torch.exp(z - torch.max(z, dim=0, keepdim=True)[0])  # for numerical stability
    return exp_z / torch.sum(exp_z, dim=0, keepdim=True)

def binarize(x):
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

def batchnorm(x, gamma, beta, running_mean, running_var, training=True, momentum=0.9, eps=1e-5):
    mu = torch.mean(x, dim=1, keepdim=True)
    var = torch.var(x, dim=1, keepdim=True, unbiased=False)
    if training:
        x_hat = (x - mu) / torch.sqrt(var + eps)
        # Update running statistics
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
    else:
        x_hat = (x - running_mean) / torch.sqrt(running_var + eps)
    cache = (x_hat, gamma, beta, mu, var, eps)
    return gamma * x_hat + beta, cache, running_mean, running_var

d, n, k = 784, 4096, 10
layer_sizes = [d, n, n, n, k]
B = 1500
L = len(layer_sizes) - 1
weights, biases, gamma, beta, running_means, running_vars = [], [], [], [], [], []
for i in range(L):
    weights.append(torch.randn(layer_sizes[i], layer_sizes[i+1], device=device) * np.sqrt(2/(layer_sizes[i] + layer_sizes[i+1])))
    biases.append(torch.randn(layer_sizes[i+1], 1, device=device) * np.sqrt(2/(layer_sizes[i] + layer_sizes[i+1])))

for i in range(L-1):
    gamma.append(torch.ones((layer_sizes[i+1], 1), device=device))
    beta.append(torch.zeros((layer_sizes[i+1], 1), device=device))
    running_means.append(torch.zeros((layer_sizes[i+1], 1), device=device))
    running_vars.append(torch.ones((layer_sizes[i+1], 1), device=device))

def feedforward(W, x, b, gamma, beta, training=True):
    h = [x]
    a = []
    caches = []
    bn_outputs = []
    for i in range(L):
        Wb = binarize(W[i])
        a.append(Wb.T @ h[-1] + b[i])
        if i == L - 1:
            h.append(softmax(a[-1]))
        else:
            h_i, cache, running_means[i], running_vars[i] = batchnorm(a[-1], gamma[i], beta[i], running_means[i], running_vars[i], training=training)
            caches.append(cache)
            bn_outputs.append(h_i)
            h.append(binarize(h_i))
    return h, a, caches, bn_outputs

def backpropagation(W, h, y, gamma, beta, caches, bn_outputs, eps=1e-5):
    batch_size = h[0].shape[1]
    grad_a = [h[-1] - y]
    grad_gamma = []
    grad_beta = []
    grad_W = [(h[-2] @ grad_a[-1].T) / batch_size]
    grad_b = [torch.sum(grad_a[-1], dim=1, keepdim=True) / batch_size]
    for i in range(L-2, -1, -1):
        Wb_next = binarize(W[i+1])
        G = (Wb_next @ grad_a[-1])
        ste_mask = (torch.abs(bn_outputs[i]) <= 1).float()
        # ste_mask = torch.ones_like(bn_outputs[i])
        G *= ste_mask
        Q = G * gamma[i]
        cache = caches[i]
        a_hat = cache[0]
        var = cache[4]
        grad_gamma.append(torch.sum(G * a_hat, dim=1, keepdim=True) / batch_size)
        grad_beta.append(torch.sum(G, dim=1, keepdim=True) / batch_size)
        grad_a.append((Q - torch.mean(Q, dim=1, keepdim=True) - a_hat * torch.mean(Q * a_hat, dim=1, keepdim=True)) / torch.sqrt(var + eps))
        grad_W.append((h[i] @ grad_a[-1].T) / batch_size)
        grad_b.append(torch.sum(grad_a[-1], dim=1, keepdim=True) / batch_size)
    grad_W.reverse()
    grad_b.reverse()
    grad_gamma.reverse()
    grad_beta.reverse()
    return grad_W, grad_b, grad_gamma, grad_beta

# MNIST Dataset
X = torch.load('train_images.pt', map_location=device)
Y = torch.load('train_labels.pt', map_location=device)

# Testing dataset
X_t = torch.load('test_images.pt', map_location=device)
Y_t = torch.load('test_labels.pt', map_location=device)

eta = 0.7
mem = 0.8
epochs = 100
v_W, v_b, v_gamma, v_beta = [torch.zeros_like(w) for w in weights], [torch.zeros_like(b) for b in biases], [torch.zeros_like(g) for g in gamma], [torch.zeros_like(b) for b in beta]
accuracy_history = []

for epoch in range(epochs):
    perm = torch.randperm(X.shape[1])
    for start in range(0, X.shape[1], B):
        end = start + B
        idx = perm[start:end]
        lookahead_W = [w - mem * v for w, v in zip(weights, v_W)]
        lookahead_b = [b - mem * v for b, v in zip(biases, v_b)]
        lookahead_gamma = [g - mem * v for g, v in zip(gamma, v_gamma)]
        lookahead_beta = [b - mem * v for b, v in zip(beta, v_beta)]
        x_batch = X[:,idx]
        y_batch = Y[:,idx]
        h, a, caches, bn_outputs = feedforward(lookahead_W, x_batch, lookahead_b, lookahead_gamma, lookahead_beta)
        grad_W, grad_b, grad_gamma, grad_beta = backpropagation(lookahead_W, h, y_batch, lookahead_gamma, lookahead_beta, caches, bn_outputs)
        for i in range(L):
            v_W[i].mul_(mem).add_(grad_W[i], alpha=eta)
            v_b[i].mul_(mem).add_(grad_b[i], alpha=eta)
            weights[i].sub_(v_W[i])
            biases[i].sub_(v_b[i])
            weights[i].clamp_(-1, 1)
            # weights[i] = torch.clamp(weights[i], -1, 1)
            if i < L - 1:
                v_gamma[i].mul_(mem).add_(grad_gamma[i], alpha=eta)
                v_beta[i].mul_(mem).add_(grad_beta[i], alpha=eta)
                gamma[i].sub_(v_gamma[i])
                beta[i].sub_(v_beta[i])

    with torch.no_grad():
        # output = feedforward(weights, X, biases, gamma, beta, training=False)[0][-1]
        # pred = torch.argmax(output, dim=0)
        # true = torch.argmax(Y, dim=0)
        # accuracy_train = (pred == true).float().mean().item()

        output = feedforward(weights, X_t, biases, gamma, beta, training=False)[0][-1]
        pred = torch.argmax(output, dim=0)
        true = torch.argmax(Y_t, dim=0)
        accuracy_test = (pred == true).float().mean().item()
        accuracy_history.append(accuracy_test)

        print(
            f'Epochs:{epoch+1}/{epochs} | '
            f'Testing Accuracy: {accuracy_test * 100:.2f}'
        )
        #     f'Training Accuracy: {accuracy_train * 100:.2f}'
        # )

np.save("BNN_accuracy.npy", np.array(accuracy_history))