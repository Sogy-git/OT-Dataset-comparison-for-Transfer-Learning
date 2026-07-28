import torch
import torch.nn as nn
from data_choice import create_data_loaders
from model import SimpleCNN
from train_func import test
import numpy as np


def extract_embeddings(type="full", batch_size=10, limit=1000, train_loader=None, target_loader=None):
    # Define lists to store the extracted embeddings
    extracted_embeddings = []  

    def hook_fn(module, input, output):
        extracted_embeddings.append(output.detach())

    # Define a what layer of the CNN to use the hooks from
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    weights = torch.load(f"simple_cnn_{type}.pth", map_location=device)

    model = SimpleCNN().to(device)
    model.load_state_dict(weights)
    hook_handle = model.fc1.register_forward_hook(hook_fn)
    test(model, device, train_loader, nn.CrossEntropyLoss())

    embeddings_train = torch.cat(extracted_embeddings, dim=0)
    extracted_embeddings.clear()  # Clear the list for the next extraction

    test(model, device, target_loader, nn.CrossEntropyLoss())

    embeddings_test = torch.cat(extracted_embeddings, dim=0)

    hook_handle.remove()

    return embeddings_train, embeddings_test


