import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# activation functions

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
round_ste = RoundSTE.apply

def hard_sigmoid(x):
    return torch.clamp((x + 1)/2, 0, 1)

def binary_tanh(x):
    return round_ste(hard_sigmoid(x)) * 2 - 1

def binary_sigmoid(x):
    return round_ste(hard_sigmoid(x))

# x = torch.linspace(-5, 5, 1000)
# plt.figure(figsize=(12, 4))
# plt.plot(x.numpy(), hard_sigmoid(x).numpy(), label='Hard Sigmoid', color='blue')
# plt.plot(x.numpy(), binary_tanh(x).numpy(), label='Binary Tanh', color='orange')
# plt.plot(x.numpy(), binary_sigmoid(x).numpy(), label='Binary Sigmoid', color='green')
# plt.title('Hard Sigmoid, Binary Tanh, and Binary Sigmoid Activation Functions ')
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.legend()
# plt.axhline(0, color='black', lw=0.5, ls='--')
# plt.axvline(0, color='black', lw=0.5, ls='--')
# plt.grid()
# plt.savefig('BNN Implementation/activation_functions.png')

# binarization function

def binarize(W, H=1.0, binary=True, stochastic=False):
    if binary:
        if stochastic:
            Wb = torch.where(torch.rand_like(W) < hard_sigmoid(W/H), H, -H)
        else:
            Wb = hard_sigmoid(W/H)
            Wb = round_ste(Wb) 
            Wb = torch.where(Wb == 1, H, -H)
    else:
        Wb = W
    return Wb

def build_layers(layer_specs):
    layers = []
    parameters = []
    last_out = None
    for spec in layer_specs:
        layer = spec.copy()
        if spec['type'] == 'linear':
            W = nn.Parameter(torch.empty(spec['out'], spec['in'], device=device))
            b = nn.Parameter(torch.zeros(spec['out'], device=device))
            H = np.sqrt(1.5 / (spec['in'] + spec['out']))
            with torch.no_grad():
                W.uniform_(-H, H)
            parameters.extend([W, b])
            layer['W'] = W
            layer['b'] = b
            layer['H'] = H
            last_out = spec['out']
        elif spec['type'] == 'conv':
            k = spec['kernel_size']
            W = nn.Parameter(torch.empty(spec['out'], spec['in'], k, k, device=device))
            b = nn.Parameter(torch.zeros(spec['out'], device=device))
            H = np.sqrt(1.5 / (spec['in'] * k * k + spec['out'] * k * k))
            with torch.no_grad():
                W.uniform_(-H, H)
            parameters.extend([W, b])
            layer['W'] = W
            layer['b'] = b
            layer['H'] = H
            last_out = spec['out']
        elif spec['type'] == 'batchnorm':
            gamma = nn.Parameter(torch.ones(last_out, device=device))
            beta = nn.Parameter(torch.zeros(last_out, device=device ))
            parameters.extend([gamma, beta])
            layer['gamma'] = gamma
            layer['beta'] = beta
            layer['running_mean'] = torch.zeros(last_out, device=device)
            layer['running_var'] = torch.ones(last_out, device=device)
        elif spec['type'] in ('activation', 'pool', 'flatten'):
            pass  # No parameters to initialize
        else:
            raise ValueError(f"Unsupported layer type: {spec['type']}")
        layers.append(layer)

    return layers, parameters

def forward(x, layers, training=True):
    for layer in layers:
        if layer['type'] == 'linear':
            Wb = binarize(layer['W'], H=layer['H'])
            x = F.linear(x, Wb, layer['b'])
        elif layer['type'] == 'conv':
            Wb = binarize(layer['W'], H=layer['H'])
            x = F.conv2d(x, Wb, layer['b'], stride=layer['stride'], padding=layer['padding'])
        elif layer['type'] == 'activation':
            x = layer['activation'](x)
        elif layer['type'] == 'pool':
            x = F.max_pool2d(x, kernel_size=layer['kernel_size'], stride=layer['stride'], padding=layer['padding'])
        elif layer['type'] == 'flatten':
            x = torch.flatten(x, start_dim=1)
        elif layer['type'] == 'batchnorm':
            x = F.batch_norm(x, running_mean=layer['running_mean'], running_var=layer['running_var'], weight=layer['gamma'], bias=layer['beta'], training=training, momentum=0.1, eps=1e-5)
    
    return x

def clip_weights(layers):
    for layer in layers:
        if layer['type'] in ('linear', 'conv'):
            with torch.no_grad():
                layer['W'].clamp_(-layer['H'], layer['H'])