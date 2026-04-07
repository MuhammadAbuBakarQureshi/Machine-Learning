import torch
from torch import nn
from torchvision import transforms

from pathlib import Path

from going_modular.going_modular import engine, data_setup, vision_transformer

def main():

    ## DATA PARAMETERS

    BATCH_SIZE = 32
    NUM_WORKERS = 0
    IMG_SIZE = 224


    ## MODEL PARAMETERS

    PATCH_SIZE = 16

    IN_CHANNELS=3
    OUT_CHANNELS=768
    EMBEDDING_DIMENSION=768
    NUM_HEADS=12
    ATTN_DROPOUT=0.0
    MLP_SIZE=3072
    DROPOUT=0.1
    TRANSFORMER_ENCODER_LAYER=12

    ## OPTIMIZER PARAMETERS

    LEARNING_RATE=0.03
    BETAS=(0.9, 0.999)
    WEIGHT_DECAY=0.3

    ## TRAINING PARAMETERS

    EPOCHS = 10

    print(f"[INFO] Epochs: {EPOCHS}")

    ## Device Agnostic code

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"[INFO] Using device: {device}")

    ## Dataset directory

    dataset_dir = Path('datasets/pizza_steak_sushi')

    train_dir = dataset_dir / 'train'
    test_dir = dataset_dir / 'test'


    ## Create transform

    manual_transform = transforms.Compose([

        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    ## create dataloaders

    print(f"[INFO] Creating dataloaders")

    train_dataloader, test_dataloader, classes = data_setup.create_dataloaders(train_dir=train_dir,
                                                                            test_dir=test_dir,
                                                                            transform=manual_transform,
                                                                            batch_size=BATCH_SIZE,
                                                                            num_workers=NUM_WORKERS)
    ## Create model

    ViT_Base = vision_transformer.ViT(patch_size=PATCH_SIZE,
                                    in_channels=IN_CHANNELS,
                                    out_channels=OUT_CHANNELS,
                                    embedding_dimension=EMBEDDING_DIMENSION,
                                    num_heads=NUM_HEADS,
                                    attn_dropout=ATTN_DROPOUT,
                                    mlp_size=MLP_SIZE,
                                    dropout=DROPOUT,
                                    transformer_encoder_layers=TRANSFORMER_ENCODER_LAYER,
                                    num_classes=len(classes))

    print(f"[INFO] Model: ViT Base")

    ## optimizer

    optimizer = torch.optim.Adam(params=ViT_Base.parameters(),
                                lr=LEARNING_RATE,
                                betas=BETAS,
                                weight_decay=WEIGHT_DECAY)

    print(f"[INFO] Adam optimizer with lr={LEARNING_RATE}")


    ## Loss function

    loss_fn = nn.CrossEntropyLoss()

    ## Training

    engine.fit_fn(model=ViT_Base,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                classes=classes,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                device=device,
                experiment_name="first_ViT",
                model_name="ViT-Base",
                extra=f"{EPOCHS}_epochs")
    
if __name__ == "__main__":

    main()