"""Implementation of a small convolutional neural network architecture for MNIST."""

import torch
from torch import nn


class TinyConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Convolutional layers
        self.features = nn.Sequential(
            # 1 channel input, 12 output channels, 3x3 kernel
            nn.Conv2d(1, 12, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 12 input channels, 12 output channels, 3x3 kernel
            nn.Conv2d(12, 12, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        # Fully connected layers
        self.classifier = nn.Sequential(
            # flattened size: 12 * 5 * 5 = 300
            nn.Linear(12 * 5 * 5, 24),
            nn.ReLU(),
            nn.Linear(24, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    net = TinyConv()
    print(net)

    test_input = torch.randn(1, 1, 28, 28)  # batch size 1, 1 channel, 28x28 input
    output = net(test_input)
    print(f"Output shape: {output.shape}")  # expected: torch.Size([1, 10])
