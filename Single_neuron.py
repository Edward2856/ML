import matplotlib.pyplot as plt
import numpy as np

x = np.array([2.5, 0.5])
y = np.array([0.9, 0.2])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

w = np.linspace(-5, 5, 1000)
b = np.linspace(-5, 5, 1000)
W, B = np.meshgrid(w, b)
y_pred = sigmoid(W * x[:, np.newaxis, np.newaxis] + B)
error = y_pred - y[:, np.newaxis, np.newaxis]
L = np.sum(error ** 2, axis=0) / 2
ax = plt.figure(1).add_subplot(111, projection='3d')
ax.plot_surface(W, B, L, cmap='turbo')
ax.set_title('Loss Function Surface')
ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('Loss Function (L)')

eta = 1
epochs = 1000
w = -2
b = 2

for epoch in range(epochs):
    ax.scatter(w, b, np.sum((sigmoid(w * x + b) - y) ** 2) / 2, color='black',marker='.', s=10, alpha=1)
    y_pred = sigmoid(w * x + b)
    error = y_pred - y
    dw = eta * np.sum(error * x * derivative_sigmoid(w * x + b))
    db = eta * np.sum(error * derivative_sigmoid(w * x + b))
    w -= dw
    b -= db

print(f"Final weights: {w}, Final bias: {b}")
plt.figure(2)
plt.scatter(x, y, color='blue', label='Data Points')
x_line = np.linspace(0, 3, 100)
y_line = sigmoid(w * x_line + b)
plt.plot(x_line, y_line, color='red', label='Decision Boundary')
plt.title('Logistic Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()