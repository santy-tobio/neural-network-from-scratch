"""
Visualization utilities for plotting results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Get style file path (one directory up from src/)
STYLE_PATH = Path(__file__).parent.parent.parent / "figs.mplstyle"


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save figure
    """
    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix (numpy array)
        class_names: Optional list of class names
        save_path: Optional path to save figure
    """
    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def visualize_sample_images(X_images, y_images, num_samples=3, figsize=(12, 4)):
    """
    Visualize sample images from the dataset.

    Args:
        X_images: Image data (n_samples, 28, 28) or (n_samples, 784)
        y_images: Labels (n_samples,)
        num_samples: Number of samples to visualize
        figsize: Figure size
    """
    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    plt.rcParams["figure.dpi"] = 300

    fig, axes = plt.subplots(1, num_samples, figsize=figsize)

    for i in range(num_samples):
        img = X_images[i].reshape(28, 28)

        if num_samples == 1:
            ax = axes
        else:
            ax = axes[i]

        ax.imshow(img, cmap="gray")
        ax.set_title(f"Sample {i+1}\nClass: {y_images[i]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
