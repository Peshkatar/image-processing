from typing import Callable

import torch
import torch.nn as nn
import torch.utils.data.dataloader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        metric: Callable[[int, int], float],
        n_epochs: int = 10,
        device: torch.device = torch.device("cpu"),
        debug: bool = False,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: torch.utils.data.DataLoader = train_loader
        self.val_loader: torch.utils.data.DataLoader = val_loader
        self.test_loader: torch.utils.data.DataLoader = test_loader
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion: nn.Module = criterion
        self.metric: Callable[[int, int], float] = metric
        self.n_epochs: int = n_epochs
        self.device: torch.device = device

    def __call__(self) -> None:
        """Train the model for n_epochs."""
        for epoch in range(self.n_epochs):
            # Train for one epoch
            epoch_loss = self.train_step(epoch)
            # Validate after each epoch
            val_accuracy = self.validate(self.val_loader)
            print(
                f"Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
            )

    def train_step(self, epoch: int) -> float:
        """Train the model for one epoch."""
        running_loss = 0.0

        for features, labels in tqdm(
            self.train_loader, desc=f"Epoch {epoch+1}/{self.n_epochs}", leave=False
        ):
            # load data into respective device
            features, labels = features.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            # this is because gradients are accumulated in PyTorch
            # so we need to zero them out at each iteration
            # if we don't do this, gradients will be accumulated to existing gradients
            # and this will lead to unexpected results
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Validate the model on the validation or test dataset."""
        self.model.eval()
        correct, total = 0, 0
        for features, labels in dataloader:
            features, labels = features.to(self.device), labels.to(self.device)
            outputs = self.model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        self.model.train()
        return self.metric(correct, total)
