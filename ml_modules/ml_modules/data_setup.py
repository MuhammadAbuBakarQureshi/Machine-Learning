from torchvision import datasets
from torch.utils.data import DataLoader, random_split

def get_dataloaders(train_dir,
                    test_dir,
                    BATCH_SIZE,
                    NUM_WORKERS,
                    weights=None,
                    test_transform=None,
                    train_transform=None):
    
    """
    
    Return:

        train_dataloader, test_dataloader, classes

    """

    ## if in future you wanna use weights you can update this function with this kinda logic

    # if transform == None:

    #     transform = weights.transforms()

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )


    classes = train_dataset.classes

    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size

    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS)

    return train_dataloader, test_dataloader, classes
