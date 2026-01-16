import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def compute_predictions(model, dataloader, device):
    """
    Returns all predictions and true labels for a given model and dataloader.
    """
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            batch_preds = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds)
            labels.extend(targets.numpy())

    return np.array(preds), np.array(labels)


def plot_confusion_matrix(y_true, y_pred, classes=None):
    """
    Plots a confusion matrix using matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    if classes:
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.show()


def print_classification_report(y_true, y_pred):
    """
    Prints precision, recall, f1-score for each class.
    """
    print(classification_report(y_true, y_pred))