import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

import matplotlib.pyplot as plt


def load_mnist_data(
    batch_size: int = 64,
    data_transforms: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_data = datasets.MNIST(
        root="data", train=True, transform=data_transforms, download=True
    )
    test_data = datasets.MNIST(
        root="data", train=False, transform=data_transforms, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, test_loader


def display_image(images: torch.Tensor) -> None:
    grid_img = utils.make_grid(images)
    plt.imshow(grid_img.permute(1, 2, 0), cmap="gray")
    plt.title("A single batch of images")
    plt.show()


if __name__ == "__main__":
    train_loader, test_loader = load_mnist_data()
