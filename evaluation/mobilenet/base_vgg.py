import torch
import torch.nn as nn
import torchvision.transforms as transforms

#set device
device = torch.device(
    'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])


class VGG16(nn.Module):

    def __init__(self, number_classes):
        super(VGG16, self).__init__()
        self.blck1  = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1),
            nn.Relu(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.blck6 = nn.Seqential(
            nn.Linear(25088, 4096),
            nn.Relu(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.Relu(inplace=True),
            nn.Dropout(0.5),
            #nn.Linear(4096,1000)
            nn.Linear(4096,number_classes)
        )

    def forward(self, input):
        out = self.blck1(input)
        out = self.blck2(out)
        out = self.blck3(out)
        out = self.blck4(out)
        out = torch.flatten(out, 1)
        out = self.blck5(out)
        out = torch.softmax(out, dim=1)
        return out

    