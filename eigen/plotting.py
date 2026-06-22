from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from torchvision.utils import make_grid


def plot_image_grid(
    images: Tensor | list[Tensor],
    grid_kws: dict[str, Any] | None = None,
    fig_kws: dict[str, Any] | None = None,
    ax_kws: dict[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    if isinstance(images, list) and not all(
        isinstance(image, torch.Tensor) for image in images
    ):
        raise TypeError("`images` must be a list of Tensors, not numpy arrays")

    grid_img = make_grid(images, **(grid_kws or {}))
    fig, ax = plt.subplots(**(fig_kws or {}))
    # C, W, H -> W, H, C
    ax.imshow(grid_img.permute(1, 2, 0))
    ax.set(**(ax_kws or {}))
    plt.show()
    return fig, ax
