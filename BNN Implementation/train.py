from mnist import *
# from cifar import *

print(f"Using device: {device}")
print(f"Device name: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

lr_start = 3e-3
lr_end = 3e-7
epochs = 100
lr_decay = (lr_end / lr_start) ** (1 / epochs)
mom = 0.9

layers, parameters = build_layers(layer_specs)

lr = lr_start
optimizer = torch.optim.Adam(parameters, lr=lr)

for epoch in range(epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        targets = F.one_hot(labels, num_classes=10).float()
        targets = targets * 2 - 1
        optimizer.zero_grad()
        outputs = forward(images, layers, training=True)
        loss = squared_hinge_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        clip_weights(layers)
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        # for images, labels in train_loader:
        #     images, labels = images.to(device), labels.to(device)
        #     output = forward(images, layers, training=False)
        #     pred = torch.argmax(output, dim=1)
        #     train_correct += (pred == labels).sum().item()
        #     train_total += labels.size(0)
        # accuracy_train = train_correct / train_total

        output = forward(X_t, layers, training=False)
        pred = torch.argmax(output, dim=1)
        accuracy_test = (pred == Y_t).float().mean().item()

    print(
        f'Epochs:{epoch+1}/{epochs} | '
        f'Testing Accuracy : {accuracy_test * 100:.2f}')
    #     f'Training Accuracy: {accuracy_train * 100:.2f}'
    # )