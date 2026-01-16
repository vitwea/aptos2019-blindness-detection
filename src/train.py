import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from src.data_loading import APTOSDataset, get_default_transforms, get_advanced_transforms
from src.sampling import get_balanced_sampler


def get_model(num_classes=5):
    """
    Returns a ResNet18 model adapted for the APTOS classification task.

    This function loads a pretrained ResNet18 model and replaces its final fully connected layer
    to match the number of classes in the APTOS dataset (default is 5).
    """
    model = resnet18(weights="IMAGENET1K_V1")  # Load pretrained ResNet18 weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace final layer for classification
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model to train.
        dataloader: DataLoader providing training data batches.
        criterion: Loss function to optimize.
        optimizer: Optimization algorithm.
        device: Device to run the training on (CPU or GPU).

    Returns:
        Average loss over the epoch.
    """
    model.train()  # Set model to training mode
    running_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)  # Move data to device

        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate loss

    return running_loss / len(dataloader)  # Return average loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on a validation or test dataset.

    Args:
        model: The neural network model to evaluate.
        dataloader: DataLoader providing evaluation data batches.
        criterion: Loss function to compute loss.
        device: Device to run the evaluation on (CPU or GPU).

    Returns:
        Tuple of average loss and accuracy over the dataset.
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  # Move data to device

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            running_loss += loss.item()  # Accumulate loss

            preds = outputs.argmax(dim=1)  # Get predicted class indices
            correct += (preds == labels).sum().item()  # Count correct predictions

    accuracy = correct / len(dataloader.dataset)  # Calculate accuracy
    return running_loss / len(dataloader), accuracy


def main():
    """
    Main training loop.

    Sets up the device, datasets, dataloaders, model, loss function, and optimizer.
    Runs training for 3 epochs, printing training and validation metrics each epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    train_dataset = APTOSDataset(
        csv_path="data/processed/train.csv",
        images_dir="data/raw/train_images",
        transform=get_advanced_transforms()  # Apply advanced augmentations for training
    )

    val_dataset = APTOSDataset(
        csv_path="data/processed/val.csv",
        images_dir="data/raw/train_images",
        transform=get_default_transforms()  # Apply basic transforms for validation
    )

    labels = train_dataset.data["diagnosis"].values  # Extract labels for balanced sampling
    sampler = get_balanced_sampler(labels)  # Create balanced sampler

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    model = get_model().to(device)  # Initialize model and move to device
    criterion = nn.CrossEntropyLoss()  # Define loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Define optimizer

    num_epochs = 3
    for epoch in range(num_epochs):  # Train for 3 epochs
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")


if __name__ == "__main__":
    main()