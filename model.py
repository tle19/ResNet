import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
from torchvision.transforms import transforms


#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    # transforms.RandomAffine(degrees=15, shear=15, scale=(0.85,1.15)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Normalization for validation
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#CIFAR-10 & CIFAR-100 dataset
# train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
# val_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
val_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1000, shuffle=False)

#model
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = activation()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity_conv = nn.Conv2d(in_channels, out_channels, 1, stride, 0)
        
    def forward(self, x):
        identity = self.identity_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.activation(out)
        return out
    
def ResNet(in_channels=3, init_padding=0, num_classes=10,  activation=nn.ReLU):
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, padding=init_padding),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2, stride=2), activation(),
        ResBlock(64, 128, 3, 1, 1, activation),
        ResBlock(128, 256, 3, 2, 1, activation),
        ResBlock(256, 512, 3, 2, 1, activation),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(512, 256),
        nn.Linear(256, num_classes)
    ) 
    return net

model = ResNet(in_channels=3, num_classes=100)
model = model.to(device)
opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
criterion = nn.CrossEntropyLoss(reduction="mean")
# scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

epochs = 100

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        opt.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print('Progress: [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Accuracy: {:.3f}% ({}/{})'.format(
            batch_idx, len(train_loader),
            100. * batch_idx / len(train_loader), 
            train_loss / (batch_idx + 1), 
            100. * correct / total, 
            correct, total))


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print('Progress: [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Accuracy: {:.3f}% ({}/{})'.format(
                batch_idx, len(val_loader),
                100. * batch_idx / len(val_loader), 
                test_loss / (batch_idx + 1), 
                100. * correct / total, 
                correct, total))

for epoch in range(epochs):
    train(epoch)
    test(epoch)
    # scheduler.step()