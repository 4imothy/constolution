import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
import torch.nn.functional as F


#set device
device = torch.device(
    'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader= torch.utils.data.DataLoader(trainset, batch_size=16,
                                        shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class AlexNet(nn.Module):
    def __init__(self, number_classes):
        super(AlexNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, 
            kernel_size=(11,11), stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256,
            kernel_size=(5,5), stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384,
            kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384,
            kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256,
            kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout2d(.5),
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(),
        )

        self.fc2 = (
            nn.Dropout2d(.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
        )

        self.fc3 = (
            nn.Dropout2d(.5),
            nn.Linear(in_features=4096, out_features=number_classes),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = x.to(device)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(x.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


net = AlexNet(10)
criterion = nn.CrossEntropyLoss()
optim = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
net.to(device)
N_CLASSES = 10

def train(epochs, train_loader, test_loader):
    net.train()
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12}")
    for epoch in range(epochs):
        e_loss = 0
        train_count = 0
        correct_train = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optim.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()
            e_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_count += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / train_count
        train_loss = e_loss / len(train_loader)
        net.eval()
        correct_test = 0
        test_count = 0
        test_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                test_loss += loss
                _, predicted = torch.max(outputs, 1)
                test_count += labels.size(0)
                correct_test += (predicted == labels).sum().item()
            test_accuracy = 100 * correct_test / test_count
            test_loss = test_loss / len(test_loader)
        epoch_str = f'{epoch+1}/{epochs}'
        print(f'{epoch_str:<6} {train_loss:<12.4f}{train_accuracy:<12.2f} {test_loss:<12.2f} {test_accuracy:<12.2f}')



    # save key metrics in json file after an epoch is done
    
    print('Finished Training')

if __name__ == "__main__":
    # download dataset here and get train and dataloaders here
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader= torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=False, num_workers=2)
    
    train(10, trainloader, testloader)