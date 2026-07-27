import torch
import torch.nn as nn
from model import SimpleCNN
from supervised_train_data import train_loader, test_loader

# Load hyperparameters and set device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
lr = 0.001
epochs = 5

# Load the model, optimizer, and loss function
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

# Training loop
def train(model, device, train_loader, optimizer, loss, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss_value.item():.6f}')

# Testing loop
def test(model, device, test_loader, loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Main training and testing loop
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, loss, epoch)
    test(model, device, test_loader, loss)

# Save the trained model
torch.save(model.state_dict(), "simple_cnn_random.pth")
