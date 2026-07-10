import torch

mnist = torch.load('mnist_accuracy.pt')
cifar = torch.load('cifar_accuracy.pt')
svhn = torch.load('svhn_accuracy.pt')

print("MNIST Accuracy:", f"{max(mnist) * 100:.2f}")
print("CIFAR Accuracy:", f"{max(cifar) * 100:.2f}")
print("SVHN Accuracy:", f"{max(svhn) * 100:.2f}")

print("\n")

print("MNIST error:", f"{(1 - max(mnist)) * 100:.2f}")
print("CIFAR error:", f"{(1 - max(cifar)) * 100:.2f}")
print("SVHN error:", f"{(1 - max(svhn)) * 100:.2f}")