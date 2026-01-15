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