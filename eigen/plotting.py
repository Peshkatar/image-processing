from this import d
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from torchvision.utils import make_grid


def plot_image_grid(
    images: Tensor | list[Tensor] | np.ndarray | list[np.ndarray],
    grid_kws: dict[str, Any] | None = None,
    fig_kws: dict[str, Any] | None = None,
    ax_kws: dict[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    elif isinstance(images, list) and all(
        isinstance(image, np.ndarray) for image in images
    ):
        images = [torch.from_numpy(x) for x in images]

    grid_img = make_grid(images, **(grid_kws or {}))
    fig, ax = plt.subplots(**(fig_kws or {}))
    ax.imshow(grid_img.permute(1, 2, 0).numpy())
    ax.set(**(ax_kws or {}))
    plt.show()
    return fig, ax
