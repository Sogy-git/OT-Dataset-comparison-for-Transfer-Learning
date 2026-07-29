import torch
from torch.utils.data import DataLoader
from data_input import mnist_train, mnist_test, semeion_train, semeion_test


def create_data_loaders(type, batch_size=10, limit=1000, datasets=None):
    if datasets is None:
        src_mnist_train = mnist_train
        src_mnist_test = mnist_test
        src_semeion_train = semeion_train
        src_semeion_test = semeion_test
    else:
        src_mnist_train, src_mnist_test, src_semeion_train, src_semeion_test = datasets

    if type == "random":
        train_indices = torch.randperm(len(src_mnist_train))[:limit]
        train_subset = torch.utils.data.Subset(src_mnist_train, train_indices)

        test_indices = torch.randperm(len(src_mnist_test))[:limit]
        test_subset = torch.utils.data.Subset(src_mnist_test, test_indices)

        mnist_train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
        mnist_test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False)

    elif type == "full":
        mnist_train_loader = DataLoader(dataset=src_mnist_train, batch_size=batch_size, shuffle=True)
        mnist_test_loader = DataLoader(dataset=src_mnist_test, batch_size=batch_size, shuffle=False)

    elif type == "KNN":
        pass  # Placeholder for KNN data loading logic

    elif type == "OT":
        pass  # Placeholder for OT data loading logic

    semeion_train_loader = DataLoader(dataset=src_semeion_train, batch_size=batch_size, shuffle=True)
    semeion_test_loader = DataLoader(dataset=src_semeion_test, batch_size=batch_size, shuffle=False)

    return mnist_train_loader, mnist_test_loader, semeion_train_loader, semeion_test_loader
