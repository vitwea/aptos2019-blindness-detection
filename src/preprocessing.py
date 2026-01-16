from torchvision import transforms


def get_default_transforms(image_size=224):
    """
    Basic transforms for baseline training.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def get_augmentation_transforms(image_size=224):
    """
    Data augmentation transforms for improving generalization.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

def get_advanced_transforms(image_size=224):
    """
    Advanced data augmentation pipeline for APTOS 2019.
    Designed to improve generalization and robustness.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05
        ),
        transforms.ToTensor(),
    ])
