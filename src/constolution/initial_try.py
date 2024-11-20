import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Define the predefined kernels
def get_predefined_kernels():
    kernels = []

    # Sharpen kernel
    sharpen_kernel = torch.tensor([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]], dtype=torch.float32)
    kernels.append(sharpen_kernel)

    # Blur kernel
    blur_kernel = (1/9) * torch.ones((3, 3), dtype=torch.float32)
    kernels.append(blur_kernel)

    # Emboss kernel
    emboss_kernel = torch.tensor([[-2, -1, 0],
                                  [-1, 1, 1],
                                  [0, 1, 2]], dtype=torch.float32)
    kernels.append(emboss_kernel)

    # Outline kernel
    outline_kernel = torch.tensor([[-1, -1, -1],
                                   [-1, 8, -1],
                                   [-1, -1, -1]], dtype=torch.float32)
    kernels.append(outline_kernel)

    # Top Sobel kernel (horizontal edges)
    sobel_top = torch.tensor([[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]], dtype=torch.float32)
    kernels.append(sobel_top)

    # Bottom Sobel kernel (horizontal edges)
    sobel_bottom = torch.tensor([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]], dtype=torch.float32)
    kernels.append(sobel_bottom)

    # Left Sobel kernel (vertical edges)
    sobel_left = torch.tensor([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]], dtype=torch.float32)
    kernels.append(sobel_left)

    # Right Sobel kernel (vertical edges)
    sobel_right = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
    kernels.append(sobel_right)

    # Identity kernel
    identity_kernel = torch.tensor([[0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]], dtype=torch.float32)
    kernels.append(identity_kernel)

    # Convert list to tensor and adjust dimensions
    kernel_tensor = torch.stack(kernels)  # Shape: (num_kernels, kernel_height, kernel_width)
    return kernel_tensor

# 2. Create the Inception-like Module
class InceptionPredefinedModule(nn.Module):
    def __init__(self):
        super(InceptionPredefinedModule, self).__init__()
        self.kernels = get_predefined_kernels()
        num_kernels = self.kernels.shape[0]

        self.fixed_convs = nn.ModuleList()
        self.adjust_convs = nn.ModuleList()

        for i in range(num_kernels):
            # Fixed convolution with predefined kernel
            conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1, bias=False)
            # Set the kernel weights and fix them
            kernel = self.kernels[i].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # Shape: (1, 3, 3, 3)
            conv.weight = nn.Parameter(kernel, requires_grad=False)
            self.fixed_convs.append(conv)

            # Learnable 1x1 convolution to adjust the output
            adjust_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=True)
            self.adjust_convs.append(adjust_conv)

    def forward(self, x):
        outputs = []
        for i in range(len(self.fixed_convs)):
            out = self.fixed_convs[i](x)
            out = self.adjust_convs[i](out)
            outputs.append(out)
        # Concatenate along the channel dimension
        out = torch.cat(outputs, dim=1)
        return out

class PredefinedKernelInceptionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(PredefinedKernelInceptionCNN, self).__init__()
        self.inception_module = InceptionPredefinedModule()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # Tunable convolutional layers
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=1)  # Output channels = num_classes

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Inception-like module with predefined kernels
        x = self.inception_module(x)
        x = self.relu(x)
        x = self.pool(x)

        # Additional tunable convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)  
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        return x

# 4. Load CIFAR-10 dataset

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 mean & std
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 mean & std
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

# 5. Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# 6. Initialize the model, criterion, and optimizer
model = PredefinedKernelInceptionCNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# 7. Training the model
num_epochs = 10

for epoch in range(num_epochs):  
    running_loss = 0.0
    total = 0
    correct = 0

    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 99:
            print('[Epoch %d, Batch %5d] loss: %.3f, Accuracy: %.2f%%' %
                  (epoch + 1, i + 1, running_loss / 100, 100 * correct / total))
            running_loss = 0.0
            total = 0
            correct = 0

print('Finished Training')

# 8. Save the trained model
PATH = './cifar_predefined_inception_cnn.pth'
torch.save(model.state_dict(), PATH)

# 9. Evaluate the model on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.2f %%' % (
    100 * correct / total))

# 10. Class-wise accuracy
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %.2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
