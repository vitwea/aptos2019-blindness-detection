import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class APTOSDataset(Dataset):
    """
    PyTorch Dataset for the APTOS 2019 Blindness Detection competition.
    Loads images and labels from the Kaggle dataset structure.
    """

    def __init__(self, csv_path, images_dir, transform=None):
        """
        Args:
            csv_path (str): Path to train.csv or test.csv
            images_dir (str): Directory containing the images
            transform (callable, optional): Optional transforms to apply
        """
        # Load the CSV file into a pandas DataFrame
        self.data = pd.read_csv(csv_path)
        # Store the directory where images are located
        self.images_dir = images_dir
        # Store the transformations to apply to each image
        self.transform = transform

        # Check that the CSV contains the 'id_code' column, which identifies each image
        if "id_code" not in self.data.columns:
            raise ValueError("CSV must contain 'id_code' column")

        # Determine if the dataset has labels (diagnosis column) or not (e.g., test set)
        self.has_labels = "diagnosis" in self.data.columns

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row of the CSV corresponding to the index
        row = self.data.iloc[idx]
        # Build the full path to the image file
        img_path = os.path.join(self.images_dir, f"{row['id_code']}.png")

        # Open the image and convert it to RGB format
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if any are provided
        if self.transform:
            image = self.transform(image)

        # If labels are available, return a tuple (image, label)
        if self.has_labels:
            label = int(row["diagnosis"])
            return image, label

        # If no labels, return only the image
        return image

def get_default_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])