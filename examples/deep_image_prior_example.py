"""Main module for training and evaluating a Deep Image Prior model on custom noisy image."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate_err_msg_format

from eigen.data import add_gaussian_noise, load_image
from eigen.metrics import mean_squared_error
from eigen.plotting import plot_image_grid
from eigen.trainer import Trainer
from models.deep_image_prior import DeepImagePrior

SIGMA = 0.5
PATH = Path("data/snail.jpg")


class CustomImageDataset(Dataset):
    def __init__(self, img: torch.Tensor, img_label: torch.Tensor):
        self.img = img
        self.img_label = img_label

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img, self.img_label


def main() -> None:
    torch.manual_seed(0)

    img = load_image(PATH)
    C, W, H = img.shape
    print("Shape: ", img.shape)
    img_noisy = torch.clip(add_gaussian_noise(img, SIGMA), 0, 1)
    # display noisy image
    plot_image_grid([img, img_noisy])

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Device: {device}")
    # model initialization + training + inference
    model = DeepImagePrior(C)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        metrics={"mse": mean_squared_error},
        n_epochs=50_000,
        device=device,
    )

    z = torch.rand(C, W, H)
    data = CustomImageDataset(z, img_noisy)
    train_loader = DataLoader(data, batch_size=1, shuffle=False)

    trainer.fit(train_loader, train_loader)

    output = model(z.to(device)).to("cpu")
    plot_image_grid([img, img_noisy, z, output.detach()], grid_kws=dict(nrow=2))

    # serialize model to onnx
    torch.onnx.export(
        model,
        (z,),
        "checkpoints/DeepImagePrior/deep_image_prior.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "width", 3: "height"},
            "output": {0: "batch_size", 2: "width", 3: "height"},
        },
    )


if __name__ == "__main__":
    main()
