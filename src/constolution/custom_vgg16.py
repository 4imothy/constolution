import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import constolution as ct
import pandas as pd
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
trainloader= torch.utils.data.DataLoader(trainset, batch_size=1024,
                                        shuffle=True, num_workers=8,  pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024,
                                        shuffle=False, num_workers=8,  pin_memory=True)



class earlyBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride, max_size = 5):
        super(earlyBlock, self).__init__()
        "kernel1, kernel2, kernel3, kernel4, kernel5"
        self.filter1 = ct.Constolution2D(ct.Kernels.Gabor, 
            input_channels, output_channels, stride=stride, spatial_size = (3,3))
        
        self.filter2 = ct.Constolution2D(ct.Kernels.Gaussian, 
            input_channels, output_channels, stride=stride, spatial_size = (3,3))
        
        self.filter3 = ct.Constolution2D(ct.Kernels.SobelHorizontalEdge, 
            input_channels, output_channels, stride=stride, spatial_size = (3,3))
        
        self.filter4 = ct.Constolution2D(ct.Kernels.SobelVerticalEdge, 
            input_channels, output_channels, stride=stride, spatial_size = (3,3))
        
        self.filter5 = ct.Constolution2D(ct.Kernels.Schmid, 
            input_channels, output_channels, stride=stride, spatial_size = (3,3))
        
        self.kernels = nn.ModuleList([self.filter1, self.filter2,self.filter3,self.filter4,self.filter5 ])

        self.max_size = max_size

    def forward(self, input):
        out = input
        for kernel in self.kernels:
            # Apply each kernel sequentially
            out = kernel(out)

        return out
    



class VGG16(nn.Module):

    def __init__(self, number_classes):
        super(VGG16, self).__init__()
        self.blck1  = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
            #predefined_block = ct.sillyBlock(in_channels = 64),

            #nn.ReLU(inplace=True),
           # nn.MaxPool2d(2, 2)
        )
        self.predefined_block = earlyBlock(input_channels = 64, output_channels=64, stride = 1, max_size=4)


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
            nn.Linear(2048, 4096),
            ##CHANGED ABOVE FOR CIFAR
            #nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,number_classes)

        )

    def forward(self, input):
        out = self.blck1(input)
        out = self.predefined_block(out)
        out = self.blck2(out)
        out = self.blck3(out)
        out = self.blck4(out)
        out = self.blck5(out)
        out = torch.flatten(out, 1)
        out = self.blck6(out)
        out = torch.softmax(out, dim=1)
        return out
