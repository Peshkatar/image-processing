"""
Implementation of a small convolutional neural network architecture for image reconstruction,
based on the Deep Image Prior paper (https://arxiv.org/abs/1711.10925).
"""

import torch
from torch import nn

from eigen.data import load_image
from eigen.plotting import plot_image_grid


class DeepImagePrior(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()

        # Convolutional layers
        self.features = nn.Sequential(
            # 1 input image, 64 output channels, 3x3 kernel, 1x1 padding
            nn.Conv2d(num_channels, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),  
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),  
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),  
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),  
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),  
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),  
            nn.ReLU(),
            nn.Conv2d(64, num_channels, (3, 3), padding=(1, 1)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


if __name__ == "__main__":
    img = load_image("data/snail.jpg")
    C, W, H = img.shape
    print("Shape: ", img.shape)

    z = torch.rand(C, W, H)
    print(z.shape)

    net = DeepImagePrior(C)
    print(net)
    output = net(z)
    print(f"Output shape: {output.shape}")
    plot_image_grid([img, z, output.detach()])
