import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data_input import build_datasets
from data_choice import create_data_loaders
from distance_choice import KNN_distance_choice, OT_distance_choice, OTDD_distance_choice
from encoder import extract_embeddings
from model import SimpleCNN
from train_func import train, test, evaluate_model


def get_device():
    return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def fine_tune_model(
    method_name,
    train_loader,
    test_loader,
    save_path,
    pretrained_path="simple_cnn_full.pth",
    device=None,
    epochs=5,
    lr=0.0001,
    verbose=True,
):
    device = device or get_device()
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    weights = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(weights)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, loss, epoch)
        if verbose:
            test(model, device, test_loader, loss)

    torch.save(model.state_dict(), save_path)
    return save_path


def evaluate_checkpoint(checkpoint_path, test_loader, device=None):
    device = device or get_device()
    model = SimpleCNN().to(device)
    weights = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(weights)
    return evaluate_model(model, device, test_loader, nn.CrossEntropyLoss())


def run_experiment(
    seed,
    semeion_train_ratio,
    output_dir,
    subset_size=5000,
    batch_size=10,
    epochs=5,
    lr=0.0001,
    pretrained_path="simple_cnn_full.pth",
    random_pretrained_path="simple_cnn_random.pth",
    verbose=True,
):
    """
    Run selection, fine-tuning, and evaluation for one seed / SEMEION-ratio combo.

    Returns a dict with metadata and per-method evaluation metrics.
    """
    import os

    device = get_device()
    datasets = build_datasets(seed=seed, semeion_train_ratio=semeion_train_ratio)
    mnist_train_loader, _, semeion_train_loader, semeion_test_loader = create_data_loaders(
        "full", batch_size=batch_size, datasets=datasets,
    )

    ratio_pct = int(semeion_train_ratio * 100)
    run_dir = os.path.join(output_dir, "models", f"seed{seed}_ratio{ratio_pct}")
    os.makedirs(run_dir, exist_ok=True)

    if verbose:
        print(
            f"\n=== seed={seed}, semeion_train_ratio={semeion_train_ratio:.0%} "
            f"({len(semeion_train_loader.dataset)} train / {len(semeion_test_loader.dataset)} test) ==="
        )

    results = {
        "seed": seed,
        "semeion_train_ratio": semeion_train_ratio,
        "semeion_train_size": len(semeion_train_loader.dataset),
        "semeion_test_size": len(semeion_test_loader.dataset),
        "subset_size": subset_size,
        "methods": {},
    }

    # Baselines (no fine-tuning on selected data)
    for method_name, checkpoint in [
        ("full_zero_shot", pretrained_path),
        ("random_zero_shot", random_pretrained_path),
    ]:
        if os.path.exists(checkpoint):
            results["methods"][method_name] = evaluate_checkpoint(
                checkpoint, semeion_test_loader, device=device,
            )
            if verbose:
                acc = results["methods"][method_name]["accuracy"]
                print(f"  {method_name}: {acc:.1%}")

    # Extract embeddings (non-shuffled so row i == dataset index i)
    embedding_train_loader = DataLoader(mnist_train_loader.dataset, batch_size=batch_size, shuffle=False)
    embedding_target_loader = DataLoader(semeion_train_loader.dataset, batch_size=batch_size, shuffle=False)

    embeddings_train, embeddings_target = extract_embeddings(
        weights_path=pretrained_path,
        train_loader=embedding_train_loader,
        target_loader=embedding_target_loader,
    )

    mnist_dataset = mnist_train_loader.dataset

    # KNN selection + fine-tune
    _, knn_indices = KNN_distance_choice(
        embeddings_train, embeddings_target, k=150, subset_size=subset_size,
    )
    knn_loader = DataLoader(Subset(mnist_dataset, knn_indices), batch_size=batch_size, shuffle=True)
    knn_path = os.path.join(run_dir, "KNN.pth")
    fine_tune_model(
        "KNN", knn_loader, semeion_test_loader, knn_path,
        pretrained_path=pretrained_path, device=device, epochs=epochs, lr=lr, verbose=verbose,
    )
    results["methods"]["KNN"] = evaluate_checkpoint(knn_path, semeion_test_loader, device=device)

    # OT selection + fine-tune
    ot_indices = OT_distance_choice(embeddings_train, embeddings_target, subset_size=subset_size)
    ot_loader = DataLoader(Subset(mnist_dataset, ot_indices), batch_size=batch_size, shuffle=True)
    ot_path = os.path.join(run_dir, "OT.pth")
    fine_tune_model(
        "OT", ot_loader, semeion_test_loader, ot_path,
        pretrained_path=pretrained_path, device=device, epochs=epochs, lr=lr, verbose=verbose,
    )
    results["methods"]["OT"] = evaluate_checkpoint(ot_path, semeion_test_loader, device=device)

    # OTDD selection + fine-tune
    train_labels = mnist_dataset.targets
    semeion_subset = semeion_train_loader.dataset
    target_labels = torch.as_tensor(semeion_subset.dataset.labels)[
        torch.as_tensor(semeion_subset.indices)
    ]
    otdd_indices = OTDD_distance_choice(
        embeddings_train=embeddings_train,
        labels_train=train_labels,
        embeddings_target=embeddings_target,
        labels_target=target_labels,
        subset_size=subset_size,
    )
    otdd_loader = DataLoader(Subset(mnist_dataset, otdd_indices), batch_size=batch_size, shuffle=True)
    otdd_path = os.path.join(run_dir, "OTDD.pth")
    fine_tune_model(
        "OTDD", otdd_loader, semeion_test_loader, otdd_path,
        pretrained_path=pretrained_path, device=device, epochs=epochs, lr=lr, verbose=verbose,
    )
    results["methods"]["OTDD"] = evaluate_checkpoint(otdd_path, semeion_test_loader, device=device)

    if verbose:
        for method in ("KNN", "OT", "OTDD"):
            acc = results["methods"][method]["accuracy"]
            print(f"  {method}: {acc:.1%}")

    return results
