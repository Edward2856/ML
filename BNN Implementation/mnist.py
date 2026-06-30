from network import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1500, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=4, pin_memory=True)
X_t, Y_t = next(iter(test_loader))
X_t = X_t.to(device)
Y_t = Y_t.to(device)

layer_specs = [
    {'type': 'flatten'},

    {'type': 'linear', 'in': 784, 'out': 4096},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'linear', 'in': 4096, 'out': 4096},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'linear', 'in': 4096, 'out': 4096},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'linear', 'in': 4096, 'out': 10},
]