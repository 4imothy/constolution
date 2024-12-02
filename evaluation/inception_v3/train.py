import os
import sys
import torch
from torch import nn, optim
import csv
import time
from ..utils import load_cifar10_data, load_caltech256_data
from . import model_with_pd, model_base

def initialize_csv_writer(csv_filename):
    file = open(csv_filename, mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Time (s)', 'Train Accuracy', 'Val Accuracy'])
    return writer

def main(base: bool, cifar: bool):
    num_epochs = 20
    image_size = 299
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if base:
        batch_size = 64
        num_classes = 10
        end = 'base'
    else:
        batch_size = 32
        num_classes = 257
        end = 'predefined'

    if cifar:
        train_loader, valid_loader = load_cifar10_data(batch_size, image_size)
    else:
        train_loader, valid_loader = load_caltech256_data(batch_size, image_size)

    model = build_model(base, num_classes, device)

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
            print(outputs.shape)
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
        if val_accuracy >= 95:
            return

    final_model_save_path = os.path.join(os.path.dirname(__file__), f'cifar10_{end}_final_model.pth')
    torch.save(model.state_dict(), final_model_save_path)
    print(f'Final model saved to {final_model_save_path}')

def build_model(base: bool, num_classes, device):
    if base:
        model = model_base.Inception3(num_classes)
        model = model.to(device)
        return model
    else:
        model = model_with_pd.Inception3(num_classes)
        model = model.to(device)
        return model

if __name__ == '__main__':
    if sys.argv[1] not in ['base', 'pd']:
        print(f'invalid model {sys.argv[1]}')
        exit(1)
    if sys.argv[2] not in ['cifar', 'caltech256']:
        print(f'invalid dataset {sys.argv[2]}')
        exit(1)
    main(sys.argv[1] == 'base', sys.argv[2] == 'cifar')
