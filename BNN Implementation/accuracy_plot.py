import torch
import matplotlib.pyplot as plt

acc = torch.load('svhn_accuracy.pt')
acc_real = torch.load('svhn_real_accuracy.pt')
x = range(1, len(acc) + 1)
plt.plot(x, [a*100 for a in acc], linewidth=2, color='blue', label='SVHN BNN')
plt.plot(x, [a*100 for a in acc_real], linewidth=2, color='red', label='SVHN Real')
plt.title('SVHN Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.legend()
plt.savefig('svhn_accuracy.png')