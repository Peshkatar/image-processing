import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim,
        criterion: nn,
        n_epochs: int = 10,
        device: str = "cpu",
        debug: bool = False,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.device = device
        self.debug = debug

    def __call__(self) -> None:
        with tqdm(range(self.n_epochs), desc="Training", unit="epoch") as epochs:
            for epoch in epochs:
                # Train for one epoch
                epoch_loss = self.train_step(epoch)
                epochs.set_postfix(
                    {"Loss": f"{epoch_loss:.4f}"}
                )  # Update progress bar with loss
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.4f}")

    def train_step(self, epoch: int) -> float:
        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch+1}/{self.n_epochs}", leave=False
        )
        running_loss = 0.0

        for batch_idx, (features, labels) in enumerate(progress_bar, 1):
            features, labels = features.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(
                Loss=f"{running_loss / batch_idx:.4f}",
                LR=f"{self.optimizer.param_groups[0]['lr']:.6f}",
            )

        if self.debug:
            print("Model is training on: ", next(self.model.parameters()).device)
            print(
                f"Batch {batch_idx}, Features: {features.shape}, Labels: {labels.shape}"
            )
            print(f"Data is loaded to {self.device}")

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def test(self):
        # Add test functionality here
        pass
