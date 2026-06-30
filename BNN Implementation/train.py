from mnist import *
# from cifar import *

print(f"Using device: {device}")
print(f"Device name: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

lr = 1
epochs = 100
mom = 0.9

layers, parameters = build_layers(layer_specs)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(parameters, lr=lr, momentum=mom, nesterov=True)

for epoch in range(epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = forward(images, layers, training=True)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        clip_weights(layers)
    
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