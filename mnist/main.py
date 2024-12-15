# TODO: add user input for model selection

import torch

from metrics import calculate_accuracy
from tinyconv import TinyConv
from trainer import Trainer
from utils import display_image, load_mnist_data

if __name__ == "__main__":
    torch.manual_seed(0)
    print("Loading data...")
    train_loader, val_loader, test_loader = load_mnist_data()
    print("Data loaded successfully!")

    # display image and label.
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print(f"Feature batch shape: {images.size()}")
    print(f"Labels batch shape: {labels.size()}")
    display_image(images)

    # model initialization + training + inference
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = TinyConv().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        metric=calculate_accuracy,
        n_epochs=10,
        device=device,
    )

    trainer()

    # inference
    test_accuracy = trainer.validate(test_loader)
    print(f"Test accuracy: {test_accuracy:.2f}")

    # serialize model
    torch.save(model.state_dict(), "checkpoints/tinyconv.pth")
