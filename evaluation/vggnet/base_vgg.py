import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
import pandas as pd
import time
#set device
device = torch.device(
    'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
)




transform = transforms.Compose(
    [transforms.Resize((227, 227)), transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader= torch.utils.data.DataLoader(trainset, batch_size=64,
                                        shuffle=True, num_workers=9)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                        shuffle=False, num_workers=9)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class VGG16(nn.Module):

    def __init__(self, number_classes):
        super(VGG16, self).__init__()
        self.blck1  = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck6 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(25088, 4096),
            #nn.Linear(2048, 4096),
            ##CHANGED ABOVE FOR CIFAR
            #nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
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

def train(model, epochs = 1):
    criterion = nn.CrossEntropyLoss()
    #increasing learning rate helped significantly
    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5 )
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay = 0.005, momentum = 0.9)  



    loop = []
    training_loss = []
    training_accuracy = []
    test_loss = []
    test_accuracy = []
    header = ["Loop", "Train Loss", "Train Acc %", "Test Loss", "Test Acc %"]
    start_time = time.time()
    for epoch in range(epochs):
        loop.append(epoch + 1)
        correct = 0
        total = 0
        running_loss = 0
        for i, data in enumerate(trainloader):
            print(f"Epoch {epoch + 1}, Batch {i + 1}/{len(trainloader)}")
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        training_loss.append(running_loss / len(trainloader))
        training_accuracy.append(100 * correct / total)

        correct = 0
        total = 0
        with torch.no_grad():
            running_loss = 0
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_loss.append(running_loss/len(testloader))
            test_accuracy.append(100 * correct / total)
    finish_time = time.time()
    duration = finish_time - start_time
    data = {header[0]: loop,header[1]: training_loss,header[2]: training_accuracy,header[3]:test_loss,header[4]:test_accuracy}
    print(pd.DataFrame(data))
    pd.DataFrame(data).to_csv('model_vgg_base.csv', index=False)
    print(f"final time:  {duration:.2f} seconds")

if __name__ == "__main__":
    model = VGG16(number_classes = 10).to(device)
    model = torch.compile(model, backend="aot_eager")
    train(model= model, epochs = 10)
    torch.save(model.state_dict(), "model_vgg_base.pth")

