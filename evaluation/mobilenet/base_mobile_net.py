import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# move to device if available
device = torch.device(
    'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])


class DepthPointWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthPointWiseConv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, 
                            kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, groups=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, n_classes):
        super(MobileNet, self).__init__()
        # first layer 
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )

        # second layer 
        self.features = nn.Sequential(
            self.features,
            
            DepthPointWiseConv(32, 64, 1),
            DepthPointWiseConv(64, 128, 2),
            DepthPointWiseConv(128, 128, 1),
            DepthPointWiseConv(128, 256, 2),
            DepthPointWiseConv(256, 256, 1),
            DepthPointWiseConv(256, 512, 2),
            
            # 5 blocks 
            DepthPointWiseConv(512, 512, 1),
            DepthPointWiseConv(512, 512, 1),
            DepthPointWiseConv(512, 512, 1),
            DepthPointWiseConv(512, 512, 1),
            DepthPointWiseConv(512, 512, 1),

            # last blocks
            DepthPointWiseConv(512, 1024, 2),
            DepthPointWiseConv(1024, 1024, 1)
        )

        # avg pooling
        self.avgpool = nn.AvgPool2d(1)
        self.classfier = nn.Sequential(
            nn.Linear(1024, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.to(device)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classfier(x)
        return x
    
net = MobileNet(10)
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
            label_vecs = torch.eye(N_CLASSES, device=device)[labels]
            loss = criterion(outputs, label_vecs)
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
                label_vecs = torch.eye(N_CLASSES)[labels.to(device)]
                loss = criterion(outputs, label_vecs)
                test_loss += loss
                _, predicted = torch.max(outputs, 1)
                test_count += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                print("correct_test", correct_test)
                print("test_loss", test_loss)
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
    trainloader= torch.utils.data.DataLoader(trainset, batch_size=16,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                            shuffle=False, num_workers=2)
    
    train(10, trainloader, testloader)
    

