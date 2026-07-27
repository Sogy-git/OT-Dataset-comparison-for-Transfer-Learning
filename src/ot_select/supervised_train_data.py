import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define a random seed for reproducibility
torch.manual_seed(42)

# Define transformations for the MNIST dataset
transform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Resize((28, 28)),
])

# Load the MNIST dataset (60,000 training images and 10,000 test images)
train_dataset = datasets.MNIST(root='./data', 
                                train=True, 
                                download=True, 
                                transform=transform)

test_dataset = datasets.MNIST(root='./data', 
                               train=False, 
                               download=True, 
                               transform=transform)

limit = 1000  # Limit the number of samples 

# Create a subset of the training dataset with randomly selected samples
train_indices = torch.randperm(len(train_dataset))[:limit]
train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

# Create a subset of the test dataset with randomly selected samples
test_indices = torch.randperm(len(test_dataset))[:limit]
test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

# Create DataLoaders for training and testing datasets
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=10,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=10,
                         shuffle=False)

