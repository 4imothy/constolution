import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision

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


class VGG16(nn.Module):

    def __init__(self, number_classes):
        super(VGG16, self).__init__()
        self.blck1  = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck6 = nn.Sequential(
            #nn.Linear(25088, 4096),
            #nn.Linear(2048, 4096),
            ##CHANGED ABOVE FOR CIFAR
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,number_classes)

        )

    def forward(self, input):
        out = self.blck1(input)
        out = self.blck2(out)
        out = self.blck3(out)
        out = self.blck4(out)
        out = self.blck5(out)
        out = torch.flatten(out, 1)
        out = self.blck6(out)
        out = torch.softmax(out, dim=1)
        return out

net = VGG16(10)
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