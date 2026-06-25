import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(784, 4096),
    nn.BatchNorm1d(4096),
    nn.Sigmoid(),

    nn.Linear(4096, 4096),
    nn.BatchNorm1d(4096),
    nn.Sigmoid(),

    nn.Linear(4096, 4096),
    nn.BatchNorm1d(4096),
    nn.Sigmoid(),

    nn.Linear(4096, 10)
).to(device)

X = torch.load("train_images.pt").T.float().to(device)
Y = torch.load("train_labels.pt").to(device)

X_t = torch.load("test_images.pt").T.float().to(device)
Y_t = torch.load("test_labels.pt").to(device)

Y = torch.argmax(Y, dim=0)
Y_t = torch.argmax(Y_t, dim=0)

B = 1500

train_loader = DataLoader(TensorDataset(X, Y), batch_size=B, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.7)

criterion = nn.CrossEntropyLoss()

epochs = 1000
accuracy_history = []
for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        train_pred = model(X).argmax(dim=1)
        accuracy_train = (train_pred == Y).float().mean().item()

        test_pred = model(X_t).argmax(dim=1)
        accuracy_test = (test_pred == Y_t).float().mean().item()
    accuracy_history.append(accuracy_test)
    print(
        f'Epochs:{epoch+1}/{epochs} | '
        f'Testing Accuracy: {accuracy_test * 100:.2f} | '
        f'Training Accuracy: {accuracy_train * 100:.2f}'
    )