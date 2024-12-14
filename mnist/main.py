# TODO: set random seed
# TODO: add user input for model selection

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from tinyconv import TinyConv
from trainer import Trainer
from utils import load_mnist_data, display_image


if __name__ == "__main__":
    print("Loading data...")
    train_loader, test_loader = load_mnist_data()
    print("Data loaded successfully!")

    # Display image and label.
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print(f"Feature batch shape: {images.size()}")
    print(f"Labels batch shape: {labels.size()}")
    display_image(images)

    # Model initialization + training + inference
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(device)
    model = TinyConv().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=10,
        device=device,
    )

    trainer()

    # Inference
    output = model(images.to(device))
    _, predicted = torch.max(output, 1)
    accuracy = (predicted == labels.to(device)).sum().item() / labels.size(0)
    print(f"Predicted: {predicted}")
    print(f"Actual: {labels}")
    print(f"Accuracy: {accuracy}")

    PATH = "checkpoints/tinyconv.pth"
    torch.save(model.state_dict(), PATH)
