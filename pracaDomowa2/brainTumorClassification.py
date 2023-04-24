from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

#DATA
transform = transforms.Compose([transforms.Resize((512,512)),transforms.Grayscale(),transforms.ToTensor()])
train_dataset = datasets.ImageFolder(root='/home/mati/studia/inteligencjaObliczeniowa/pracaDomowa2/data/brainTumorData/Training',transform=transform)
test_dataset = datasets.ImageFolder(root='/home/mati/studia/inteligencjaObliczeniowa/pracaDomowa2/data/brainTumorData/Testing',transform=transform)
trainDataLoader = torch.utils.data.DataLoader(train_dataset,batch_size=6,shuffle=True)
testDataLoader = torch.utils.data.DataLoader(test_dataset,batch_size=6,shuffle=True)
class tumorClassification(nn.Module):
    def __init__(self):
        super(tumorClassification, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 5904384)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#SETTINGS
torch.cuda.empty_cache()
learning_rate = 0.03
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10
num_classes = 4
batch_size=6
criterion = nn.CrossEntropyLoss()
classes = train_dataset.classes
total_step = len(trainDataLoader)
model = tumorClassification().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_total_steps = len(trainDataLoader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainDataLoader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in testDataLoader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')