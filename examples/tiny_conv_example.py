"""Main module for training and evaluating a TinyConv model on the MNIST dataset."""

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from eigen.data import load_dataset
from eigen.metrics import accuracy
from eigen.plotting import plot_image_grid
from eigen.trainer import Trainer
from models.tiny_conv import TinyConv


def main() -> None:
    torch.manual_seed(0)
    print("Loading data...")
    train_data = MNIST(
        root="data", train=True, download=True, transform=Compose([ToTensor()])
    )
    test_data = MNIST(
        root="data", train=False, download=True, transform=Compose([ToTensor()])
    )
    train_loader, test_loader, val_loader = load_dataset(
        train_data, test_data, val_split=0.1
    )
    print("Data loaded successfully!")

    # display image and label.
    images, labels = next(iter(train_loader))
    print(
        f"Feature batch shape: {images.size()}",
        f"Labels batch shape: {labels.size()}",
        sep="\n",
    )
    plot_image_grid(images, ax_kws=dict(title="A single batch of images"))

    # model initialization + training + inference
    model = TinyConv()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Device: {device}")

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        metrics={"accuracy": accuracy},
        n_epochs=10,
        device=device,
    )

    trainer.fit(train_loader, val_loader)

    # inference
    test_accuracy = trainer.eval(test_loader)
    print(f"Test accuracy: {test_accuracy['accuracy']}%")

    # serialize model to onnx
    torch.onnx.export(
        model,
        images.to(device),
        "checkpoints/TinyConv/tinyconv.onnx",
        input_names=["input"],
        output_names=["output"],
    )


if __name__ == "__main__":
    main()
