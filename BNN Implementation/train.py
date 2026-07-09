from mnist import *
# from cifar import *
# from svhn import *
# from setproctitle import setproctitle
import os

# setproctitle("mnist_bnn_2000_batch_size")

print(f"PID: {os.getpid()}")
print(f"Using device: {device}")
print(f"Device name: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

layers, parameters = build_layers(layer_specs)

lr = lr_start
optimizer = torch.optim.Adam(parameters, lr=lr)
best_accuracy = 0.0
accuracy_history = []

for epoch in range(epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels % 10  # Ensure labels are in the range [0, 9]
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
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        # for images, labels in train_loader:
        #     images, labels = images.to(device), labels.to(device)
        #     output = forward(images, layers, training=False)
        #     pred = torch.argmax(output, dim=1)
        #     train_correct += (pred == labels).sum().item()
        #     train_total += labels.size(0)
        # accuracy_train = train_correct / train_total

        # for images, labels in test_loader:
        #     images, labels = images.to(device), labels.to(device)
        #     labels = labels % 10  # Ensure labels are in the range [0, 9]
        #     output = forward(images, layers, training=False)
        #     pred = torch.argmax(output, dim=1)
        #     test_correct += (pred == labels).sum().item()
        #     test_total += labels.size(0)
        # accuracy_test = test_correct / test_total if test_total > 0 else 0

        output = forward(X_t, layers, training=False)
        pred = torch.argmax(output, dim=1)
        accuracy_test = (pred == Y_t).float().mean().item()

        best_accuracy = max(best_accuracy, accuracy_test)
        accuracy_history.append(accuracy_test)
    print(
        f'Epochs:{epoch+1}/{epochs} | '
        f'Testing Accuracy : {accuracy_test * 100:.2f} | '
        f'Best Accuracy so far: {best_accuracy * 100:.2f}')
    #     f'Training Accuracy: {accuracy_train * 100:.2f}'
    # )

# torch.save(accuracy_history, 'mnist_accuracy.pt')