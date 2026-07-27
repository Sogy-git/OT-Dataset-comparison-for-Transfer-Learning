import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.flatten = nn.Flatten()

        # Define the convolutional layers and fully connected layers
        self.conv1stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1), # Input channels = 1 for grayscale images, Output channels = 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2stack = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # Input channels = 32 from previous layer, Output channels = 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # Fully connected layer with 128 neurons
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # Fully connected layer with 10 output classes

    def forward(self, x):
        x = self.conv1stack(x)
        x = self.conv2stack(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        return x

