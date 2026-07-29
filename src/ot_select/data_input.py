import torch
from torchvision import datasets, transforms

DEFAULT_SEED = 69
DEFAULT_SEMEION_TRAIN_RATIO = 0.1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Resize((28, 28)),
])


def build_datasets(seed=DEFAULT_SEED, semeion_train_ratio=DEFAULT_SEMEION_TRAIN_RATIO, data_root="./data"):
    """Build MNIST and SEMEION datasets with a reproducible SEMEION train/test split."""
    torch.manual_seed(seed)

    mnist_train = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform,
    )
    mnist_test = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform,
    )
    semeion_dataset = datasets.SEMEION(
        root=data_root, download=True, transform=transform,
    )

    train_size = int(semeion_train_ratio * len(semeion_dataset))
    test_size = len(semeion_dataset) - train_size
    semeion_train, semeion_test = torch.utils.data.random_split(
        semeion_dataset, [train_size, test_size],
    )

    return mnist_train, mnist_test, semeion_train, semeion_test


# Default module-level datasets (backward compatible with existing scripts)
mnist_train, mnist_test, semeion_train, semeion_test = build_datasets(
    seed=DEFAULT_SEED,
    semeion_train_ratio=DEFAULT_SEMEION_TRAIN_RATIO,
)
