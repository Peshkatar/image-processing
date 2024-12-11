import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from miniconv import MiniConvNet
from trainer import Trainer
from utils import load_mnist_data, display_image


if __name__ == "__main__":
    print("Loading data...")
    train_loader, test_loader = load_mnist_data()
    print("Data loaded successfully!")

    # Display image and label.
    train_features, train_labels = next(iter(train_loader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    # display_image(img, label)

    # Model initialization + training + inference
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = MiniConvNet().to(device)
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
    output = model(train_features[0].unsqueeze(0).to(device))
    _, predicted = torch.max(output, 1)
    print(f"Predicted: {predicted.item()}")
    print(f"Actual: {train_labels[0].item()}")
