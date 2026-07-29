import torch
import torch.nn as nn
from model import SimpleCNN
from train_func import test


def extract_embeddings(
    train_loader=None,
    target_loader=None,
    type="full",
    weights_path=None,
    batch_size=10,
    limit=1000,
):
    extracted_embeddings = []

    def hook_fn(module, input, output):
        extracted_embeddings.append(output.detach())

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    checkpoint = weights_path or f"simple_cnn_{type}.pth"
    weights = torch.load(checkpoint, map_location=device)

    model = SimpleCNN().to(device)
    model.load_state_dict(weights)
    hook_handle = model.fc1.register_forward_hook(hook_fn)

    test(model, device, train_loader, nn.CrossEntropyLoss())
    embeddings_train = torch.cat(extracted_embeddings, dim=0)
    extracted_embeddings.clear()

    test(model, device, target_loader, nn.CrossEntropyLoss())
    embeddings_test = torch.cat(extracted_embeddings, dim=0)

    hook_handle.remove()
    return embeddings_train, embeddings_test
