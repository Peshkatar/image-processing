import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms


def load_mnist_data(
    batch_size: int = 64,
    train_split: float = 0.9,
    val_split: float = 0.1,
    data_transforms: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Load MNIST dataset and return train, validation, and test data loaders."""
    if not (0 <= train_split <= 1) or not (0 <= val_split <= 1):
        raise ValueError("train_split and val_split must be between 0 and 1")
    if train_split + val_split != 1:
        raise ValueError("train_split + val_split must be equal to 1")

    train_data = datasets.MNIST(
        root="data", train=True, transform=data_transforms, download=True
    )
    test_data = datasets.MNIST(
        root="data", train=False, transform=data_transforms, download=True
    )

    # Split train_data into training and validation sets
    train_data, val_data = random_split(train_data, [train_split, val_split])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    _ = load_mnist_data()
