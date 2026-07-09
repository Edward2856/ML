import torch
import matplotlib.pyplot as plt

acc = torch.load('cifar_accuracy.pt')
acc_real = torch.load('cifar_real_accuracy.pt')
x = range(1, len(acc) + 1)
plt.plot(x, [a*100 for a in acc], linewidth=2, color='blue', label='CIFAR-10 BNN')
plt.plot(x, [a*100 for a in acc_real], linewidth=2, color='red', label='CIFAR-10 Real')
plt.title('CIFAR-10 Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.legend()
plt.savefig('cifar_accuracy.png')