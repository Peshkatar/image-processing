import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt


def load_mnist_data(
    batch_size: int = 64,
    data_transforms: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
) -> tuple[DataLoader, DataLoader]:
    train_data = MNIST(
        root="data", train=True, transform=data_transforms, download=True
    )
    test_data = MNIST(
        root="data", train=False, transform=data_transforms, download=True
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, test_loader


def display_image(img: torch.Tensor, label: torch.Tensor) -> None:
    plt.imshow(img, cmap="gray")
    plt.title(label)
    plt.show()


if __name__ == "__main__":
    train_loader, test_loader = load_mnist_data()
