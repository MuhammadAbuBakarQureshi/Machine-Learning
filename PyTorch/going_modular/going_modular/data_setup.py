from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int
):
    
    """

    Get training and testing dataloaders

    Returns:
        train_dataloader, test_dataloader, classes

    """

    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transform)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform)

    classes =  train_data.classes

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    return train_dataloader, test_dataloader, classes
