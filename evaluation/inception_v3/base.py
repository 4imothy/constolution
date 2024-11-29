import os
import torch
from torch import nn, optim
from torch import amp
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import csv
import time

BASE = True

def initialize_csv_writer(csv_filename):
    file = open(csv_filename, mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Time (s)', 'Train Accuracy', 'Val Accuracy'])
    return writer

def main():
    num_epochs = 600
    batch_size = 256
    image_size = 299
    num_classes = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, valid_loader = load_imagenet_data(batch_size, image_size)
    model = build_model(num_classes, device)
    writer = initialize_csv_writer(os.path.join(os.path.dirname(__file__), 'base.csv'))

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)
    scaler = amp.grad_scaler.GradScaler('cuda')

    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            with amp.autocast_mode.autocast('cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")
        train_accuracy = 100. * correct / total

        model.eval()
        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in valid_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)

                _, pred_top1 = outputs.topk(1, dim=1, largest=True, sorted=True)
                _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)

                top1_correct += (pred_top1.squeeze() == targets).sum().item()
                top5_correct += (pred_top5 == targets.view(-1, 1)).sum().item()
                total += targets.size(0)

        top1_accuracy = 100 * top1_correct / total
        top5_accuracy = 100 * top5_correct / total
        print(f"Validation Accuracy: Top-1: {top1_accuracy:.2f}%, Top-5: {top5_accuracy:.2f}%")
        scheduler.step()
        elapsed_time = time.time() - start_time
        writer.writerow([epoch + 1, elapsed_time, train_accuracy, top1_accuracy])

def load_imagenet_data(batch_size, image_size):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pfsdir = os.getenv('PFSDIR')
    data_root = os.path.join(pfsdir, 'data') if pfsdir else './data'
    train_dataset = torchvision.datasets.ImageNet(
        root=data_root, split='train', transform=train_transforms, download=True
    )
    val_dataset = torchvision.datasets.ImageNet(
        root=data_root, split='val', transform=val_transforms, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader

def build_model(num_classes, device):
    if BASE:
        model = torchvision.models.inception_v3(pretrained=False)
        assert model.fc.out_features == num_classes
        model = model.to(device)
        return model
    # TODO the customized model
    model = torchvision.models.inception_v3(pretrained=False)
    assert model.fc.out_features == num_classes
    model = model.to(device)
    return model

if __name__ == "__main__":
    main()
