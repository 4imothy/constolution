import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# move to device if available
device = torch.device(
    'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])


class DepthPointWiseConv(nn.module):
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


class MobileNet(nn.module):
    def __init__(self, n_classes):
        
        # first layer 
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.reLU(inplace=True),
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
            nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classfier(x)
        return x
    
net = MobileNet(100)
criterion = nn.CrossEntropyLoss()
optim = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
net.to(device)

def train(epochs, trainloader):
    
    for epoch in range(epochs):  

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data

            optim.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()
            
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        # save key metrics in json file after an epoch is done
    
    print('Finished Training')

if __name__ == "__main__":
    # download dataset here and get train and dataloaders here

    pass

