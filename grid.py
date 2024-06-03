import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as O
from sklearn.model_selection import ParameterGrid

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

#normalization for validation
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#CIFAR-10 & CIFAR-100 dataset
# train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
# val_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
val_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=60, shuffle=True)
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
        nn.LazyLinear(num_classes)
    ) 
    return net

model = ResNet(in_channels=3, num_classes=100)
model = model.to(device)

epochs = 2

#hyperparameters grid
param_grid = {
    'lr': [0.001, 0.002, 0.005],
    'weight_decay': [0, 0.0001, 0.0005, 0.001],
    'batch_size': [128, 256, 512]
}

best_accuracy = 0
best_params = None

#grid search
for params in ParameterGrid(param_grid):
    print("Training with parameters:", params)
    
    #hyperparameters
    lr = params['lr']
    weight_decay = params['weight_decay']
    batch_size = params['batch_size']
    
    #optimizer and criterion
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    
    #training
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
        
        #validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"EPOCH {epoch + 1}: Validation Accuracy: {accuracy}")
    
    #best hyperparameters
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

print("Best validation accuracy:", best_accuracy)
print("Best hyperparameters:", best_params)