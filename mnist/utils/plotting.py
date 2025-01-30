import matplotlib.pyplot as plt
import torch
from torchvision import utils


def display_image(images: torch.Tensor) -> None:
    grid_img = utils.make_grid(images)
    plt.imshow(grid_img.permute(1, 2, 0), cmap="gray")
    plt.title("A single batch of images")
    plt.show()
