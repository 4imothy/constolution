import os
import sys
import torch
import constolution as ct
from torch import nn
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

N_EPOCH = 15
TRAIN = 'train'
TEST = 'test'
LR = 0.01
MOMENTUM = 0.9
N_CLASSES = 10

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = ct.Constolution2D(ct.Kernels.Gaussian, 3, 32, 5)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.log_softmax(x, dim=1)


def train(model, train_loader, test_loader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12}")

    for epoch in range(N_EPOCH):
        e_loss = 0
        train_count = 0
        correct_train = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            label_vecs = torch.eye(N_CLASSES)[labels]
            loss = criterion(outputs, label_vecs)
            loss.backward()
            optimizer.step()
            e_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_count += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / train_count
        train_loss = e_loss / len(train_loader)
        model.eval()
        correct_test = 0
        test_count = 0
        test_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                label_vecs = torch.eye(N_CLASSES)[labels]
                loss = criterion(outputs, label_vecs)
                test_loss += loss
                _, predicted = torch.max(outputs, 1)
                test_count += labels.size(0)
                correct_test += (predicted == labels).sum().item()

            test_accuracy = 100 * correct_test / test_count
            test_loss = test_loss / len(test_loader)
        epoch_str = f'{epoch+1}/{N_EPOCH}'
        print(f'{epoch_str:<6} {train_loss:<12.4f} {train_accuracy:<12.2f} {test_loss:<12.2f} {test_accuracy:<12.2f}')


if __name__ == "__main__":
    args = sys.argv[1:]
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')
    classes = datasets.CIFAR10(root='./datasets', download=True, transform=transforms.ToTensor()).classes

    train_dataset = datasets.CIFAR10(
        root='./datasets', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10(
        root='./datasets', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = Model()
    train(model, train_loader, test_loader)
    torch.save(model.state_dict(), model_path)
