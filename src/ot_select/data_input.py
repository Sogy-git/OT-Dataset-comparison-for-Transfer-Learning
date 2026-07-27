import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define a random seed for reproducibility
torch.manual_seed(42)

# Define global transformation
transform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Resize((28, 28)),
])

# Load the MNIST dataset (60,000 training images and 10,000 test images)
mnist_train = datasets.MNIST(root='./data', 
                                train=True, 
                                download=True, 
                                transform=transform)

mnist_test = datasets.MNIST(root='./data', 
                               train=False, 
                               download=True, 
                               transform=transform)

# Load the SEMEION dataset (1,000 training images and 1,000 test images)
semeion_dataset = datasets.SEMEION(root='./data',
                                        download=True,
                                        transform=transform)

# Split the SEMEION dataset into training and testing sets (10% training, 90% testing)
train_size = int(0.1 * len(semeion_dataset))
test_size = len(semeion_dataset) - train_size

semeion_train, semeion_test = torch.utils.data.random_split(
    semeion_dataset,
    [train_size, test_size])
