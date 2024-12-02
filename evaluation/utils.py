import os
import torchvision
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STDDEV = [0.229, 0.224, 0.225]

def load_cifar10_data(batch_size, image_size):
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ])

    pfsdir = os.getenv('PFSDIR')
    data_root = os.path.join(pfsdir, 'datasets') if pfsdir else './datasets'
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, transform=train_transforms, download=True
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, transform=val_transforms, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.transform(image)
        target = one_hot_encode(label)
        return image, target

def to_rgb(x):
    return x.convert('RGB')

def one_hot_encode(target, num_classes=257):
    one_hot = torch.zeros(num_classes)
    one_hot[target] = 1
    return one_hot

def load_caltech256_data(batch_size, image_size, split_ratio=0.8):
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDDEV),
    ])

    pfsdir = os.getenv('PFSDIR')
    data_root = os.path.join(pfsdir, 'datasets') if pfsdir else './datasets'

    full_dataset = torchvision.datasets.Caltech256(
        root=data_root, download=True
    )

    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset = TransformDataset(train_dataset, train_transforms)
    val_dataset = TransformDataset(val_dataset, val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
