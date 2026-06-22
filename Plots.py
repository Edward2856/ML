import numpy as np
import matplotlib.pyplot as plt

bnn_acc = np.load("BNN_accuracy.npy")
real_acc = np.load("Real_weighted_model_accuracy.npy")

bnn_acc *= 100
real_acc *= 100

epochs_bnn = range(1, len(bnn_acc) + 1)
epochs_real = range(1, len(real_acc) + 1)

plt.plot(epochs_bnn, bnn_acc, label='BNN (2 layers, 324 Neurons each, BN + sign)', color='blue', marker='o', lw=2.5, ms=5)
plt.plot(epochs_real, real_acc, label='Real (2 layers, 1024 Neurons each, BN + sigmoid)', color='red', marker='o', lw=2.5, ms=5)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()
plt.grid()
plt.savefig('accuracy_comparison.png')