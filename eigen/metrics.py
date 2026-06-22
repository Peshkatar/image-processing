"""Metrics module for calculating model performance statistics."""

import torch


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    _, predicted = torch.max(outputs, 1)
    return 100 * (predicted == labels).sum().item() / labels.size(0)
