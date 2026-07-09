# from network import *
from network_real import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=4, pin_memory=True)
X_t, Y_t = next(iter(test_loader))
X_t = X_t.to(device)
Y_t = Y_t.to(device)

lr_start = 3e-3
lr_end = 3e-7
epochs = 1000
lr_decay = (lr_end / lr_start) ** (1 / epochs)
activation = F.relu

# layer_specs = [
#     {'type': 'flatten'},
#     {'type': 'dropout', 'p': 0.2},

#     {'type': 'linear', 'in': 784, 'out': 4096},
#     {'type': 'batchnorm'},
#     {'type': 'activation', 'activation': activation},
#     {'type': 'dropout', 'p': 0.5},

#     {'type': 'linear', 'in': 4096, 'out': 4096},
#     {'type': 'batchnorm'},
#     {'type': 'activation', 'activation': activation},
#     {'type': 'dropout', 'p': 0.5},

#     {'type': 'linear', 'in': 4096, 'out': 4096},
#     {'type': 'batchnorm'},
#     {'type': 'activation', 'activation': activation},
#     {'type': 'dropout', 'p': 0.5},

#     {'type': 'linear', 'in': 4096, 'out': 10},
#     {'type': 'batchnorm'},
# ]

layer_specs = [
    {'type': 'conv', 'in': 1, 'out': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': activation},

    {'type': 'conv', 'in': 128, 'out': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'pool', 'kernel_size': 2, 'stride': 2, 'padding': 0},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': activation},

    {'type': 'conv', 'in': 128, 'out': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': activation},

    {'type': 'conv', 'in': 256, 'out': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'pool', 'kernel_size': 2, 'stride': 2, 'padding': 0},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': activation},

    {'type': 'conv', 'in': 256, 'out': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': activation},

    {'type': 'conv', 'in': 512, 'out': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'pool', 'kernel_size': 2, 'stride': 2, 'padding': 0},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': activation},

    {'type': 'flatten'},

    {'type': 'linear', 'in': 512 * 3 * 3, 'out': 1024},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': activation},

    {'type': 'linear', 'in': 1024, 'out': 1024},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': activation},
    
    {'type': 'linear', 'in': 1024, 'out': 10},
    # {'type': 'batchnorm'},
]