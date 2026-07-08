import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = grad_output.clone()

        # derivative of hard sigmoid
        grad[x.abs() > 1] = 0

        return grad

binarize = SignSTE.apply
binary_tanh = SignSTE.apply

def build_layers(layer_specs):
    layers = []
    parameters = []
    last_out = None
    for spec in layer_specs:
        layer = spec.copy()
        if spec['type'] == 'linear':
            W = nn.Parameter(torch.empty(spec['out'], spec['in'], device=device))
            limit = np.sqrt(1.5 / (spec['in'] + spec['out']))
            # lr_scale = 1.0 / limit
            with torch.no_grad():
                W.uniform_(-limit, limit)
            parameters.append(W)
            layer['W'] = W
            layer['b'] = None
            layer['H'] = 1.0
            # layer['lr_scale'] = lr_scale
            last_out = spec['out']
        elif spec['type'] == 'conv':
            k = spec['kernel_size']
            W = nn.Parameter(torch.empty(spec['out'], spec['in'], k, k, device=device))
            limit = np.sqrt(1.5 / (spec['in'] * k * k + spec['out'] * k * k))
            # lr_scale = 1.0 / limit
            with torch.no_grad():
                W.uniform_(-limit, limit)
            parameters.append(W)
            layer['W'] = W
            layer['b'] = None
            # layer['lr_scale'] = lr_scale
            layer['H'] = 1.0
            last_out = spec['out']
        elif spec['type'] == 'batchnorm':
            gamma = nn.Parameter(torch.ones(last_out, device=device))
            beta = nn.Parameter(torch.zeros(last_out, device=device ))
            parameters.extend([gamma, beta])
            layer['gamma'] = gamma
            layer['beta'] = beta
            layer['running_mean'] = torch.zeros(last_out, device=device)
            layer['running_var'] = torch.ones(last_out, device=device)
        elif spec['type'] in ('activation', 'pool', 'flatten', 'dropout'):
            pass 
        else:
            raise ValueError(f"Unsupported layer type: {spec['type']}")
        layers.append(layer)

    return layers, parameters

def forward(x, layers, training=True):
    for layer in layers:
        if layer['type'] == 'linear':
            Wb = binarize(layer['W'])
            x = F.linear(x, Wb, layer['b'])
        elif layer['type'] == 'conv':
            Wb = binarize(layer['W'])
            x = F.conv2d(x, Wb, layer['b'], stride=layer['stride'], padding=layer['padding'])
        elif layer['type'] == 'activation':
            x = layer['activation'](x)
        elif layer['type'] == 'pool':
            x = F.max_pool2d(x, kernel_size=layer['kernel_size'], stride=layer['stride'], padding=layer['padding'])
        elif layer['type'] == 'flatten':
            x = torch.flatten(x, start_dim=1)
        elif layer['type'] == 'batchnorm':
            x = F.batch_norm(x, running_mean=layer['running_mean'], running_var=layer['running_var'], weight=layer['gamma'], bias=layer['beta'], training=training, momentum=0.1, eps=1e-4)
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