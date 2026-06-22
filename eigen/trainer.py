from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainingMetrics:
    loss: float
    metrics: dict[str, float]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], float]],
        n_epochs: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.n_epochs = n_epochs
        self.device = device

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callback: Callable[[TrainingMetrics], None] | None = None,
    ) -> list[TrainingMetrics]:
        metrics_history = []

        for epoch in range(self.n_epochs):
            epoch_loss = self._train_epoch(train_loader, epoch)
            metrics = self.eval(val_loader)

            metrics = TrainingMetrics(loss=epoch_loss, metrics=metrics)
            metrics_history.append(metrics)

            if callback:
                callback(metrics)
                continue

            print(
                f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {metrics.loss:.4f}, , ".join(
                    f"{metric}: {value:.2f}"
                    for metric, value in metrics.metrics.items()
                )
            )

        return metrics_history

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0

        for features, labels in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{self.n_epochs}"
        ):
            loss = self._train_step(features, labels)
            running_loss += loss

        return running_loss / len(dataloader)

    def _train_step(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        features = features.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.inference_mode()
    def eval(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_outputs = []
        all_labels = []

        for features, labels in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(features)
            all_outputs.append(outputs)
            all_labels.append(labels)

        outputs = torch.cat(all_outputs)
        labels = torch.cat(all_labels)

        return {
            name: metric_fn(outputs, labels) for name, metric_fn in self.metrics.items()
        }
