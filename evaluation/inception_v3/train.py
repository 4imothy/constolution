import os
import torch
from torch import nn, optim
import torchvision
import csv
import time
from ..utils import load_cifar10_data

BASE = True

def initialize_csv_writer(csv_filename):
    file = open(csv_filename, mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Time (s)', 'Train Accuracy', 'Val Accuracy'])
    return writer

def main():
    num_epochs = 100
    batch_size = 128
    image_size = 299
    num_classes = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, valid_loader = load_cifar10_data(batch_size, image_size)
    model = build_model(num_classes, device)
    if BASE:
        end = 'base'
    else:
        end = 'predefined'
    writer = initialize_csv_writer(os.path.join(os.path.dirname(__file__), f'cifar10_{end}.csv'))

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

    start_time = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(train_loader)
        print(f'Training Loss: {avg_loss:.4f}')
        train_accuracy = 100. * correct / total

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in valid_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_accuracy = 100. * correct / total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        elapsed_time = time.time() - start_time
        writer.writerow([epoch + 1, elapsed_time, train_accuracy, val_accuracy])
        if val_accuracy >= 0.95:
            return

def build_model(num_classes, device):
    if BASE:
        model = torchvision.models.inception_v3(init_weights=True)
        model.aux_logits = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        return model
    else:
        print('predef not defined')
        exit(1)

if __name__ == '__main__':
    main()
