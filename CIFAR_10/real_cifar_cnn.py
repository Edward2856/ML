import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

conv_weights = [
    torch.randn((32, 3, 3, 3), device=device) * np.sqrt(2/(3*3*3 + 32*3*3)),
    torch.randn((64, 32, 3, 3), device=device) * np.sqrt(2/(3*3*32 + 64*3*3)),
]
conv_weights[0].requires_grad = True
conv_weights[1].requires_grad = True

conv_biases = [
    torch.zeros(32, device=device, requires_grad=True),
    torch.zeros(64, device=device, requires_grad=True),
]

fc_weights = [
    torch.randn(4096, 512, device=device) * np.sqrt(2/(4096)),
    torch.randn(512, 10, device=device) * np.sqrt(2/(512)),
]
fc_weights[0].requires_grad = True
fc_weights[1].requires_grad = True

fc_biases = [
    torch.zeros(512, 1, device=device, requires_grad=True),
    torch.zeros(10, 1, device=device, requires_grad=True),
]

conv_gamma = [
    torch.ones(1, 32, 1, 1, device=device, requires_grad=True),
    torch.ones(1, 64, 1, 1, device=device, requires_grad=True),
]
conv_beta = [
    torch.zeros(1, 32, 1, 1, device=device, requires_grad=True),
    torch.zeros(1, 64, 1, 1, device=device, requires_grad=True),
]

conv_running_means = [
    torch.zeros(1, 32, 1, 1, device=device),
    torch.zeros(1, 64, 1, 1, device=device),
]

conv_running_vars = [
    torch.ones(1, 32, 1, 1, device=device),
    torch.ones(1, 64, 1, 1, device=device),
]

fc_gamma = [torch.ones(512, 1, device=device, requires_grad=True)]
fc_beta = [torch.zeros(512, 1, device=device, requires_grad=True)]
fc_running_means = [torch.zeros(512, 1, device=device)]
fc_running_vars = [torch.ones(512, 1, device=device)]

parameters = conv_weights + conv_biases + conv_gamma + conv_beta + fc_weights + fc_biases + fc_gamma + fc_beta

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)
X_t, Y_t = next(iter(test_loader))
X_t = X_t.to(device)
Y_t = Y_t.to(device)

epochs = 200
eta = 0.7
mem = 0.8
optimizer = torch.optim.Adam(parameters, lr=eta)
# v_conv_weights, v_conv_biases, v_conv_gamma, v_conv_beta = [torch.zeros_like(w) for w in conv_weights], [torch.zeros_like(b) for b in conv_biases], [torch.zeros_like(g) for g in conv_gamma], [torch.zeros_like(b) for b in conv_beta]
# v_fc_weights, v_fc_biases, v_fc_gamma, v_fc_beta = [torch.zeros_like(w) for w in fc_weights], [torch.zeros_like(b) for b in fc_biases], [torch.zeros_like(g) for g in fc_gamma], [torch.zeros_like(b) for b in fc_beta] 

def batchnorm(x, gamma, beta, running_mean, running_var, training=True, momentum=0.9, eps=1e-5):
    mu = torch.mean(x, dim=1, keepdim=True)
    var = torch.var(x, dim=1, keepdim=True, unbiased=False)
    if training:
        x_hat = (x - mu) / torch.sqrt(var + eps)
        with torch.no_grad():
            running_mean.mul_(momentum).add_(mu, alpha=1 - momentum)
            running_var.mul_(momentum).add_(var, alpha=1 - momentum)
    else:
        x_hat = (x - running_mean) / torch.sqrt(running_var + eps)
    # cache = (x_hat, gamma, beta, mu, var, eps)
    return gamma * x_hat + beta, running_mean, running_var

def batchnorm_conv(x, gamma, beta, running_mean, running_var, training=True, momentum=0.9, eps=1e-5):
    mu = torch.mean(x, dim=(0, 2, 3), keepdim=True)
    var = torch.var(x, dim=(0, 2, 3), keepdim=True, unbiased=False)
    if training:
        x_hat = (x - mu) / torch.sqrt(var + eps)
        with torch.no_grad():
            running_mean.mul_(momentum).add_(mu, alpha=1 - momentum)
            running_var.mul_(momentum).add_(var, alpha=1 - momentum)
    else:
        x_hat = (x - running_mean) / torch.sqrt(running_var + eps)
    # cache = (x_hat, gamma, beta, mu, var, eps)
    return gamma * x_hat + beta, running_mean, running_var

def feedforward(x, training=True):
    # h = []

    # conv1
    x = F.conv2d(x, conv_weights[0], bias=conv_biases[0], stride=1, padding=1)
    x, conv_running_means[0], conv_running_vars[0] = batchnorm_conv(x, conv_gamma[0], conv_beta[0], conv_running_means[0], conv_running_vars[0], training=training)
    x = F.relu(x)
    # h.append(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # conv2
    x = F.conv2d(x, conv_weights[1], bias=conv_biases[1], stride=1, padding=1)
    x, conv_running_means[1], conv_running_vars[1] = batchnorm_conv(x, conv_gamma[1], conv_beta[1], conv_running_means[1], conv_running_vars[1], training=training)
    x = F.relu(x)
    # h.append(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # flatten
    x = torch.flatten(x, start_dim=1)

    # fc1
    x = fc_weights[0].T @ x.T + fc_biases[0]
    x, fc_running_means[0], fc_running_vars[0] = batchnorm(x, fc_gamma[0], fc_beta[0], fc_running_means[0], fc_running_vars[0], training=training)
    x = F.relu(x)

    # fc2
    x = fc_weights[1].T @ x + fc_biases[1]

    return x

for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        output = feedforward(x_batch, training=True)
        loss = F.cross_entropy(output.T, y_batch)
        loss.backward()
        optimizer.step()
    
    train_correct = 0
    test_correct = 0
    total_train = 0
    total_test = 0
    with torch.inference_mode():
        # for x_train, y_train in train_loader:
        #     x_train = x_train.to(device)
        #     y_train = y_train.to(device)
        #     output = feedforward(x_train, training=False)
        #     pred = torch.argmax(output, dim=0)
        #     train_correct += (pred == y_train).sum().item()
        #     total_train += y_train.size(0)
        # accuracy_train = train_correct / total_train
        
        output = feedforward(X_t, training=False)
        pred = torch.argmax(output.T, dim=1)
        accuracy_test = (pred == Y_t).float().mean().item()
    print(f'Epochs:{epoch+1}/{epochs} | Test Accuracy: {accuracy_test * 100:.2f}%')