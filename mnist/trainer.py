"""Training utilities for PyTorch models with metrics tracking."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    epoch_loss: float
    accuracy: float


class Trainer:
    """Class for training and evaluating a model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        metric: Callable[[int, int], float],
        n_epochs: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.n_epochs = n_epochs
        self.device = device
        self.model.to(self.device)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callback: Callable[[TrainingMetrics], None] | None = None,
    ) -> list[TrainingMetrics]:
        """Train the model and return metrics history."""
        metrics_history = []

        for epoch in range(self.n_epochs):
            epoch_loss = self._train_epoch(train_loader, epoch)
            accuracy = self.validate(val_loader)

            metrics = TrainingMetrics(epoch_loss=epoch_loss, accuracy=accuracy)
            metrics_history.append(metrics)

            if callback:
                callback(metrics)
            else:
                print(
                    f"Epoch {epoch + 1}/{self.n_epochs}, "
                    f"Loss: {metrics.epoch_loss:.4f}, "
                    f"Val Accuracy: {metrics.accuracy:.2f}%"
                )

        return metrics_history

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0

        for features, labels in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{self.n_epochs}"
        ):
            loss = self._train_step(features, labels)
            running_loss += loss

        return running_loss / len(dataloader)

    def _train_step(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Execute single training step."""
        
        features = features.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.inference_mode()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model on the validation or test dataset."""
        self.model.eval()
        correct, total = 0, 0

        for features, labels in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return self.metric(correct, total)
