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


class VGGNet(nn.Module):

    def __init__(self):
        pass

    def forward():
        pass

    