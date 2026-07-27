import torch
import torch.nn as nn
from model import SimpleCNN
from data_choice import create_data_loaders
from train_func import train, test, trainloop

# Load hyperparameters and set device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
lr = 0.001
epochs = 5
batch_size = 10
batch_size_full = 64
limit = 1000
types = ["random", "full"]

# Train and test the model for each data type on MNIST dataset
for type in types:
    print(f"Training with {type} data:")
    if type == "full":
        train_loader, test_loader, _, _ = create_data_loaders(type, batch_size=batch_size_full)
    else: 
        train_loader, test_loader, _, _ = create_data_loaders(type, batch_size=batch_size, limit=limit)
    trainloop(type, device, lr, epochs, train_loader, test_loader)

# Test the trained models on SEMEION dataset
for type in types:
    print(f"Testing with {type} data:")
    weights = torch.load(f"simple_cnn_{type}.pth", map_location=device)
    _, _, _, test_loader = create_data_loaders(type)
    model = SimpleCNN().to(device)
    model.load_state_dict(weights)
    test(model, device, test_loader, nn.CrossEntropyLoss())