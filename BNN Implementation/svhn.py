from network import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4371, 0.4438, 0.4728),
        std=(0.1980, 0.2010, 0.1970)
        )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4371, 0.4438, 0.4728),
        std=(0.1980, 0.2010, 0.1970)
        )
])

train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=train_transform)
extra = datasets.SVHN(root='./data', split='extra', download=True, transform=train_transform)
train_dataset = torch.utils.data.ConcatDataset([train_dataset, extra])
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)
# X_t, Y_t = next(iter(test_loader))
# X_t = X_t.to(device)
# Y_t = (Y_t % 10).to(device)  # Ensure labels are in the range [0, 9]

lr_start = 1e-3
lr_end = 1e-6
epochs = 200
lr_decay = (lr_end / lr_start) ** (1 / epochs)

layer_specs = [
    {'type': 'conv', 'in': 3, 'out': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'conv', 'in': 64, 'out': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'type': 'pool', 'kernel_size': 2, 'stride': 2, 'padding': 0},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'conv', 'in': 64, 'out': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
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

    {'type': 'flatten'},
    {'type': 'linear', 'in': 256 * 4 * 4, 'out': 1024},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'linear', 'in': 1024, 'out': 1024},
    {'type': 'batchnorm'},
    {'type': 'activation', 'activation': binary_tanh},

    {'type': 'linear', 'in': 1024, 'out': 10},
    {'type': 'batchnorm'},
]