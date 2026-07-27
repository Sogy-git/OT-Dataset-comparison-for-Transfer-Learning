import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data_input import mnist_train, mnist_test, semeion_train, semeion_test

def create_data_loaders(type, batch_size=10, limit=1000):

    if type == "random":
        # Create a subset of the training dataset with randomly selected samples
        train_indices = torch.randperm(len(mnist_train))[:limit]
        train_subset = torch.utils.data.Subset(mnist_train, train_indices)

        # Create a subset of the test dataset with randomly selected samples
        test_indices = torch.randperm(len(mnist_test))[:limit]
        test_subset = torch.utils.data.Subset(mnist_test, test_indices)

        # Create DataLoaders for training and testing datasets
        mnist_train_loader = DataLoader(dataset=train_subset,
                                  batch_size=batch_size,
                                  shuffle=True)

        mnist_test_loader = DataLoader(dataset=test_subset,
                                 batch_size=batch_size,
                                 shuffle=False)

    elif type == "full":
        mnist_train_loader = DataLoader(dataset=mnist_train,
                                  batch_size=batch_size,
                                  shuffle=True)

        mnist_test_loader = DataLoader(dataset=mnist_test,
                                 batch_size=batch_size,
                                 shuffle=False)

    elif type =='KNN':
        pass  # Placeholder for KNN data loading logic

    elif type == 'OT':
        pass  # Placeholder for OT data loading logic

    semeion_train_loader = DataLoader(dataset=semeion_train,
                                      batch_size=batch_size,
                                      shuffle=True)
    semeion_test_loader = DataLoader(dataset=semeion_test,
                                     batch_size=batch_size,
                                     shuffle=False)
    
    return mnist_train_loader, mnist_test_loader, semeion_train_loader, semeion_test_loader
