from network import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=4, pin_memory=True)
X_t, Y_t = next(iter(test_loader))
X_t = X_t.to(device)
Y_t = Y_t.to(device)

layer_specs = [
    {'type': 'conv', 'in': 3, 'out': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'conv', 'in': 128, 'out': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'pool', 'kernel_size': 2, 'stride': 2, 'padding': 0},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'conv', 'in': 128, 'out': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'conv', 'in': 256, 'out': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'pool', 'kernel_size': 2, 'stride': 2, 'padding': 0},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'conv', 'in': 256, 'out': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'conv', 'in': 512, 'out': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'pool', 'kernel_size': 2, 'stride': 2, 'padding': 0},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'flatten'},
    {'type': 'linear', 'in': 512 * 4 * 4, 'out': 1024},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'linear', 'in': 1024, 'out': 1024},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'linear', 'in': 1024, 'out': 10},
    {'type': 'batchnorm'},
]