from torch.utils.data import WeightedRandomSampler
from torchvision import datasets
from torch.utils.data import DataLoader
from collections import Counter

def get_dataloaders(train_dir,
                    test_dir,
                    BATCH_SIZE,
                    NUM_WORKERS,
                    create_sampler=False,
                    shuffle=True,
                    test_transform=None,
                    train_transform=None):
    
    """
    
    Return:

        train_dataloader, test_dataloader, classes

    """

    ## Datasets

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )

    classes = train_dataset.classes
    
    ## Sampler

    if create_sampler:

        class_count = Counter(train_dataset.targets)

        class_weights = [ 1.0 / class_count[y] for y in train_dataset.targets ]

        sampler = WeightedRandomSampler(
            weights= class_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

        
        ## Dataloader

        train_dataloader = DataLoader(train_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            sampler=sampler,
                            shuffle=shuffle)

    else:

        train_dataloader = DataLoader(train_dataset,
                                    batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKERS,
                                    shuffle=shuffle)

    test_dataloader = DataLoader(test_dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS)

    return train_dataloader, test_dataloader, classes
