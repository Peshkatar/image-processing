import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: use nn.Sequential to simplify the code
class TinyConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 1 channel input, 12 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 12, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        # 12 input channels, 12 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(12, 12, (3, 3))
        self.pool2 = nn.MaxPool2d((2, 2))
        # flattened size: 12 * 5 * 5 = 300
        self.fc = nn.Linear(12 * 5 * 5, 24)
        self.out = nn.Linear(24, 10)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = F.relu(self.conv1(X))
        X = self.pool1(X)
        X = F.relu(self.conv2(X))
        X = self.pool2(X)
        # Flatten
        X = torch.flatten(X, 1)  # flatten all dimensions except batch
        X = F.relu(self.fc(X))
        # linear output layer (10 classes)
        X = self.out(X)
        return X


if __name__ == "__main__":
    net = TinyConv()
    print(net)

    X = torch.randn(1, 1, 28, 28)  # batch size 1, 1 channel, 28x28 input
    output = net(X)
    print(f"Output shape: {output.shape}")  # expected: torch.Size([1, 10])
