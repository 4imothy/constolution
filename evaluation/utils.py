import os
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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
