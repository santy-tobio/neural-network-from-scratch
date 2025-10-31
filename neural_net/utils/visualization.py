from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.typing import RcStyleType
from numpy.typing import NDArray


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 4),
    show: bool = True,
    style: RcStyleType | None = None,
) -> Figure:
    """
    Plot training and validation loss/accuracy.
    """
    # Use custom style if available
    if style:
        plt.style.use(style)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Loss
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(
        epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3
    )
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy (if available) - check for both "train_acc" and "train_metric"
    has_accuracy = False
    train_key = None
    val_key = None

    if "train_acc" in history and "val_acc" in history:
        train_key = "train_acc"
        val_key = "val_acc"
        has_accuracy = True
    elif "train_metric" in history and "val_metric" in history:
        train_key = "train_metric"
        val_key = "val_metric"
        has_accuracy = True

    if has_accuracy:
        ax2.plot(
            epochs, history[train_key], label="Train Acc", marker="o", markersize=3
        )
        ax2.plot(epochs, history[val_key], label="Val Acc", marker="s", markersize=3)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "Accuracy not tracked",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Training and Validation Accuracy")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
    cm: NDArray[np.int_],
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
    show: bool = True,
    style: RcStyleType | None = None,
) -> Figure:
    """
    Plot confusion matrix as a heatmap.
    """
    # Use custom style if available
    if style:
        plt.style.use(style)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else range(cm.shape[1]),
        yticklabels=class_names if class_names else range(cm.shape[0]),
        cbar_kws={"label": "Count"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_one_per_class(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    save_path: str | Path = "figures/one_per_class.png",
    figsize: tuple[int, int] = (16, 12),
    show: bool = True,
    style: RcStyleType | None = None,
) -> Figure:
    """
    Visualize one image per class in a grid.
    """
    # Use custom style if available
    if style:
        plt.style.use(style)
    plt.rcParams["figure.dpi"] = 300

    # Ensure figures directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Get unique classes
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # Create grid (7x7 = 49, good for 47 classes)
    n_rows = 7
    n_cols = 7

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Plot one image per class
    for idx, class_label in enumerate(unique_classes):
        # Find first occurrence of this class
        class_indices = np.where(y == class_label)[0]
        sample_idx = class_indices[0]

        # Get image
        img = X[sample_idx].reshape(28, 28)

        # Plot
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(f"Class {int(class_label)}", fontsize=8)
        axes[idx].axis("off")

    # Hide unused subplots
    for idx in range(n_classes, n_rows * n_cols):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_class_distribution(
    y: NDArray[np.integer],
    save_path: str | Path = "figures/class_distribution.png",
    figsize: tuple[int, int] = (14, 6),
    show: bool = True,
    style: RcStyleType | None = None,
) -> Figure:
    """
    Plot the distribution of classes in the dataset.
    """

    # Use custom style if available
    if style:
        plt.style.use(style)
    plt.rcParams["figure.dpi"] = 300

    # Ensure figures directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Count samples per class
    unique_classes, counts = np.unique(y, return_counts=True)

    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        unique_classes,
        counts,
        color="#4165c0",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Class Label", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Class Distribution in EMNIST Dataset", fontsize=14)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")
    if show:
        plt.show()

    return fig


def plot_robustness_curves(
    robustness_results: dict,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 6),
    show: bool = True,
    style: RcStyleType | None = None,
) -> Figure:
    """
    Plot robustness curves showing accuracy degradation under noise.

    Args:
        robustness_results: Dict with results for each model
        save_path: Path to save the figure
        figsize: Figure size
        show: Whether to display the plot
        style: Matplotlib style to use

    Returns:
        matplotlib Figure object
    """
    # Use custom style if available
    if style:
        plt.style.use(style)

    fig, ax = plt.subplots(figsize=figsize)

    # Colors for each model (you can customize these)
    colors = plt.cm.tab10(np.linspace(0, 1, len(robustness_results)))

    # Plot accuracy vs noise level for each model
    for idx, (model_name, results) in enumerate(robustness_results.items()):
        noise_lvls = [r["noise_level"] for r in results]
        accuracies = [r["accuracy"] for r in results]

        ax.plot(
            noise_lvls,
            accuracies,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=model_name,
            color=colors[idx],
        )

    ax.set_xlabel("Noise Level (σ)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model Robustness: Accuracy vs Noise Level", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(-0.02, max([r["noise_level"] for r in list(robustness_results.values())[0]]) + 0.02)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Robustness plot saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_noisy_samples(
    X_test: NDArray[np.floating],
    y_test: NDArray[np.integer],
    noise_levels: list[float],
    num_samples: int = 5,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 10),
    show: bool = True,
    style: RcStyleType | None = None,
) -> Figure:
    """
    Visualize sample images with different noise levels.

    Args:
        X_test: Test images (flattened)
        y_test: Test labels
        noise_levels: List of noise levels to visualize
        num_samples: Number of samples to show
        save_path: Path to save the figure
        figsize: Figure size
        show: Whether to display the plot
        style: Matplotlib style to use

    Returns:
        matplotlib Figure object
    """
    # Use custom style if available
    if style:
        plt.style.use(style)

    # Select random samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)

    X_samples = X_test[sample_indices]
    y_samples = y_test[sample_indices]

    fig, axes = plt.subplots(num_samples, len(noise_levels), figsize=figsize)

    for i in range(num_samples):
        X_img = X_samples[i].reshape(28, 28)
        label = int(y_samples[i])

        for j, noise_level in enumerate(noise_levels):
            ax = axes[i, j]

            # Add noise
            if noise_level > 0:
                np.random.seed(42 + i)
                noise = np.random.normal(0, noise_level, X_img.shape)
                X_noisy = np.clip(X_img + noise, 0, 1)
            else:
                X_noisy = X_img

            # Plot
            ax.imshow(X_noisy, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

            # Titles
            if i == 0:
                ax.set_title(f"σ={noise_level:.1f}", fontsize=12, fontweight="bold")
            if j == 0:
                ax.text(
                    -10,
                    14,
                    f"Label: {label}",
                    fontsize=10,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontweight="bold",
                )

    plt.suptitle(
        "Effect of Gaussian Noise on Test Samples", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Noisy samples visualization saved to {save_path}")

    if show:
        plt.show()

    return fig
