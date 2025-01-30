"""Main module for training and evaluating a TinyConv model on the MNIST dataset."""

# TODO: add user input for model selection

import torch
from metrics import calculate_accuracy
from tinyconv import TinyConv
from trainer import Trainer
from utils.data import load_mnist_data
from utils.plotting import display_image

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
    model = TinyConv()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        metric=calculate_accuracy,
        n_epochs=10,
        device=device,
    )

    trainer.fit(train_loader, val_loader)

    # inference
    test_accuracy = trainer.validate(test_loader)
    print(f"Test accuracy: {test_accuracy:.2f}")

    # serialize model to onnx
    torch.onnx.export(
        model,
        images.to(device),
        "checkpoints/tinyconv.onnx",
        input_names=["input"],
        output_names=["output"],
    )
