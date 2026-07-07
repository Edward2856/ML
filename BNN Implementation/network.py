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

class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, H):
        ctx.save_for_backward(W)
        ctx.H = H

        out = torch.clamp((W / H + 1) / 2, 0, 1)
        out = torch.round(out)
        out = out * 2 - 1
        out = out * H

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (W,) = ctx.saved_tensors
        H = ctx.H

        grad = grad_output.clone()
        grad *= 0.5

        # derivative of hard sigmoid
        grad[torch.abs(W / H) > 1] = 0

        return grad, None

binarize = BinarizeSTE.apply

def hard_sigmoid(x):
    return torch.clamp((x + 1)/2, 0, 1)

def binary_tanh(x):
    return round_ste(hard_sigmoid(x)) * 2 - 1

# def binary_sigmoid(x):
#     return round_ste(hard_sigmoid(x), H=1.0)

# binarization function

# def binarize(W, H=1.0, binary=True, stochastic=False):
#     if binary:
#         if stochastic:
#             Wb = torch.where(torch.rand_like(W) < hard_sigmoid(W/H), H, -H)
#         else:
#             Wb = hard_sigmoid(W/H)
#             Wb = round_ste(Wb) 
#             Wb = Wb * 2 - 1
#             Wb = Wb * H
#             # Wb = torch.where(Wb == 1, H, -H)    
#     else:
#         Wb = W
#     return Wb

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
        # elif spec['type'] == 'batchnorm':
        #     gamma = nn.Parameter(torch.ones(last_out, device=device))
        #     beta = nn.Parameter(torch.zeros(last_out, device=device ))
        #     parameters.extend([gamma, beta])
        #     layer['gamma'] = gamma
        #     layer['beta'] = beta
        #     layer['running_mean'] = torch.zeros(last_out, device=device)
        #     layer['running_var'] = torch.ones(last_out, device=device)
        elif spec['type'] == 'batchnorm1d':
            bn = nn.BatchNorm1d(last_out, eps=1e-4, momentum=0.1).to(device)
            parameters.extend(list(bn.parameters()))
            layer['bn'] = bn
        elif spec['type'] == 'batchnorm2d':
            bn = nn.BatchNorm2d(last_out, eps=1e-4, momentum=0.1).to(device)
            parameters.extend(list(bn.parameters()))
            layer['bn'] = bn
        elif spec['type'] in ('activation', 'pool', 'flatten', 'dropout'):
            pass  # No parameters to initialize
        else:
            raise ValueError(f"Unsupported layer type: {spec['type']}")
        layers.append(layer)

    return layers, parameters

def forward(x, layers, training=True):
    for layer in layers:
        if layer['type'] == 'linear':
            Wb = binarize(layer['W'], layer['H'])
            x = F.linear(x, Wb, layer['b'])
        elif layer['type'] == 'conv':
            Wb = binarize(layer['W'], layer['H'])
            x = F.conv2d(x, Wb, layer['b'], stride=layer['stride'], padding=layer['padding'])
        elif layer['type'] == 'activation':
            x = layer['activation'](x)
        elif layer['type'] == 'pool':
            x = F.max_pool2d(x, kernel_size=layer['kernel_size'], stride=layer['stride'], padding=layer['padding'])
        elif layer['type'] == 'flatten':
            x = torch.flatten(x, start_dim=1)
        # elif layer['type'] == 'batchnorm':
        #     x = F.batch_norm(x, running_mean=layer['running_mean'], running_var=layer['running_var'], weight=layer['gamma'], bias=layer['beta'], training=training, momentum=0.9, eps=1e-4)
        elif layer['type'] in ('batchnorm1d', 'batchnorm2d'):
            if training:
                layer['bn'].train()
            else:
                layer['bn'].eval()
            x = layer['bn'](x)
        elif layer['type'] == 'dropout':
            if training:
                x = F.dropout(x, p=layer['p'], training=True)
        else:
            raise ValueError(f"Unsupported layer type: {layer['type']}")
    return x

def clip_weights(layers):
    for layer in layers:
        if layer['type'] in ('linear', 'conv'):
            with torch.no_grad():
                layer['W'].clamp_(-layer['H'], layer['H'])

def squared_hinge_loss(outputs, targets):
    return torch.mean(torch.clamp(1 - outputs * targets, min=0) ** 2)