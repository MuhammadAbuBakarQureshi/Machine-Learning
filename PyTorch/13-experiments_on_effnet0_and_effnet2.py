
import torch
from torch import nn
from torchvision import transforms
from timeit import default_timer as timer
from pathlib import Path


from going_modular.going_modular import download_data, data_setup, utils, engine

def main():

    ## Device agnostic code

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device

    print(f"[INFO] Using device: {device}")

    ## DATA creation

    ## Download data

    download_data.download_data(root="datasets/pizza_steak_sushi_20_percent",
                                source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")

    ## Data path

    data_10_percent_path = Path('datasets/pizza_steak_sushi')
    data_20_percent_path = Path('datasets/pizza_steak_sushi_20_percent/pizza_steak_sushi')

    train_dir_10_percent = data_10_percent_path / 'train'
    train_dir_20_percent = data_20_percent_path / 'train'

    test_dir_10_percent = data_10_percent_path / 'test'

    ## Create Dataloaders

    print(f"[INFO] Creating dataloaders")

    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )

    data_transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        normalize
    ])

    BATCH_SIZE = 32
    NUM_WORKERS = 0


    train_dataloader_10_percent, test_dataloader_10_percent, classes = data_setup.create_dataloaders(train_dir=train_dir_10_percent,
                                test_dir=test_dir_10_percent,
                                transform=data_transform,
                                num_workers=NUM_WORKERS,
                                batch_size=BATCH_SIZE)

    train_dataloader_20_percent, test_dataloader_10_percent, classes = data_setup.create_dataloaders(train_dir=train_dir_20_percent,
                                test_dir=test_dir_10_percent,
                                transform=data_transform,
                                num_workers=NUM_WORKERS,
                                batch_size=BATCH_SIZE)

    train_dataloader_10_percent, train_dataloader_20_percent, test_dataloader_10_percent, classes

    ## Experminet parameters


    num_epochs = [5, 10]

    models = ['effnet_b0', 'effnet_b2']

    train_dataloaders = {
        'data_10_percent': train_dataloader_10_percent,
        'data_20_percent': train_dataloader_20_percent
    }


    start_time = timer()

    experiments = 0

    for epochs in num_epochs:

        for dataloader_name, dataloader in train_dataloaders.items():

            for model_name in models:

                experiments += 1
                print(f"[INFO] Experminet number: {experiments}")
                print(f"[INFO] Model: {model_name}")
                print(f"[INFO] Dataloader: {dataloader_name}")
                print(f"[INFO] Number of epochs: {epochs}")

                if model_name == "effnet_b0":
                    model = utils.create_effnetb0()
                elif model_name == "effnet_b2":
                    model = utils.create_effnetb2()

                ## Create loss function and optimizer

                loss_fn = nn.CrossEntropyLoss()

                optimizer = torch.optim.Adam(params=model.parameters(),
                                        lr=0.001)

                engine.fit_fn(model=model,
                            train_dataloader=dataloader,
                            test_dataloader=test_dataloader_10_percent,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            classes=classes,
                            batch_size=BATCH_SIZE,
                            epochs=epochs,
                            device=device,
                            experiment_name=dataloader_name,
                            model_name=model_name,
                            extra=f"{epochs}_epochs")
                
                MODEL_SAVE_PATH = f"models"
                MODEL_NAME = f"{experiments+9}_{model_name}_{dataloader_name}_{epochs}_epochs"

                utils.save_model(model=model,
                                target_dir=MODEL_SAVE_PATH,
                                model_name=MODEL_NAME)
                
                print("\n\n" + "-"*60 + "\n\n")

    end_time = timer()

    utils.print_train_time(start=start_time,
                        end=end_time,
                        device=device)
    
if __name__ == "__main__":

    main()