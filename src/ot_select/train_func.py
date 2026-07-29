import torch
import torch.nn as nn
from model import SimpleCNN


def train(model, device, train_loader, optimizer, loss, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss_value.item():.6f}"
            )


def evaluate_model(model, device, test_loader, loss):
    """Evaluate a model and return structured metrics."""
    model.eval()
    test_loss = 0.0
    correct = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            predictions.extend(pred.view(-1).cpu().tolist())
            targets.extend(target.cpu().tolist())

    total = len(test_loader.dataset)
    avg_loss = test_loss / total
    accuracy = correct / total

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
        "targets": targets,
    }


def test(model, device, test_loader, loss):
    metrics = evaluate_model(model, device, test_loader, loss)
    print(
        f"\nTest set: Average loss: {metrics['loss']:.4f}, "
        f"Accuracy: {metrics['correct']}/{metrics['total']} "
        f"({100. * metrics['accuracy']:.0f}%)\n"
    )
    return metrics


def trainloop(type, device, lr, epoch, train_loader, test_loader):
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(1, epoch + 1):
        train(model, device, train_loader, optimizer, loss, epoch)
        test(model, device, test_loader, loss)

    torch.save(model.state_dict(), f"simple_cnn_{type}.pth")
