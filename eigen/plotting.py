from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from torchvision.utils import make_grid


def display_image_grid(
    images: Tensor, ax_kws: dict[str, Any] | None = None
) -> tuple[Figure, Axes]:
    grid_img = make_grid(images)
    fig, ax = plt.subplots(**(ax_kws or {}))
    ax.imshow(grid_img.permute(1, 2, 0), cmap="gray")
    ax.set(title="A single batch of images")
    plt.show()
    return fig, ax
