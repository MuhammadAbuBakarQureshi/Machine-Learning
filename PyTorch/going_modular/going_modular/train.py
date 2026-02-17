
from download_data import download_data
from data_setup import create_dataloaders
from model_builder import TinyVGG
from engine import fit_fn
from utils import save_model

import torch
from torch import nn
from torchvision import transforms
from torchmetrics import Accuracy
from pathlib import Path

BATCH_SIZE=32
NUM_WORKERS=1
EPOCHS=2
HIDDEN_UNITS=10
LEARNING_RATE=0.01


def main():

    # setup directories
    data_path = Path("../datasets")
    train_dir = data_path / "pizza_steak_sushi/train"
    test_dir = data_path / "pizza_steak_sushi/test"


    # device agnostic code

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # 1. Download data

    print("Downloading data...")
    download_data(root="../datasets")

    # 2. Data setup

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    print("Creating dataloaders...")
    train_dataloader, test_dataloader, classes = create_dataloaders(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    transform=data_transform,
                                                                    batch_size=BATCH_SIZE,
                                                                    num_workers=NUM_WORKERS)

    # 3. Model

    print("Model creation...")
    model = TinyVGG(input_shape=3,
                    hidden_units=HIDDEN_UNITS,
                    output_shape=len(classes))

    # 4. Model training

    train_accuracy_fn = Accuracy(task='multiclass', num_classes=len(classes)).to(device)
    test_accuracy_fn = Accuracy(task='multiclass', num_classes=len(classes)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=LEARNING_RATE)

    fit_fn(model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_accuracy_fn=train_accuracy_fn,
        test_accuracy_fn=test_accuracy_fn,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        summary_write_path="../runs/model_1/",
        device=device)

    save_model(model=model,
            target_dir="../models",
            model_name="11-PyTorch-Going-Modular-Script-Mode-TinyVGG-Model.pt")
    

if __name__ == "__main__":
    main()