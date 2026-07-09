import torch
import matplotlib.pyplot as plt

acc = torch.load('mnist_conv_accuracy.pt')
acc_real = torch.load('mnist_real_conv_accuracy.pt')
x = range(1, len(acc) + 1)
plt.plot(x, [a*100 for a in acc], linewidth=2, color='blue', label='MNIST BNN')
plt.plot(x, [a*100 for a in acc_real], linewidth=2, color='red', label='MNIST Real')
plt.title('MNIST (Convolutional) Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.legend()
plt.savefig('mnist_conv_accuracy.png')