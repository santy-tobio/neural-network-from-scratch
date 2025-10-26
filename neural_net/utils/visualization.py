"""Visualization utilities for plotting results with proper type hints."""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Get style file path (one directory up from src/)
STYLE_PATH = Path(__file__).parent.parent.parent / "figs.mplstyle"


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 4),
    show: bool = True,
) -> Figure:
    """
    Plot training and validation loss/accuracy.
    """
    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

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

    # Accuracy (if available)
    if "train_acc" in history and "val_acc" in history:
        ax2.plot(
            epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3
        )
        ax2.plot(epochs, history["val_acc"], label="Val Acc", marker="s", markersize=3)
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
) -> Figure:
    """
    Plot confusion matrix as a heatmap.
    """
    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

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

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {save_path}")

    if show:
        plt.show()

    return fig


def visualize_sample_images(
    X_images: NDArray[np.floating],
    y_images: NDArray[np.integer],
    num_samples: int = 3,
    figsize: tuple[int, int] = (12, 4),
    image_shape: tuple[int, int] = (28, 28),
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """
    Visualize sample images from the dataset.

    """
    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    fig, axes = plt.subplots(1, num_samples, figsize=figsize)

    for i in range(num_samples):
        # Reshape if needed
        img = X_images[i].reshape(image_shape) if X_images[i].ndim == 1 else X_images[i]

        ax = axes if num_samples == 1 else axes[i]

        ax.imshow(img, cmap="gray")
        ax.set_title(f"Sample {i+1}\nClass: {y_images[i]}")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metrics: list[str],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """
    Plot a bar chart comparing multiple models on different metrics.

    """
    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    model_names = list(results.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Extract metric values
        values = [results[model].get(metric, 0) for model in model_names]

        # Create bar plot
        bars = ax.bar(range(len(model_names)), values, alpha=0.8)

        # Color bars by performance
        colors = cm.RdYlGn(
            np.array(values) / max(values) if max(values) > 0 else [0.5] * len(values)
        )
        for bar, color in zip(bars, colors, strict=False):
            bar.set_color(color)

        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for _, (bar, value) in enumerate(zip(bars, values, strict=False)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_noise_robustness(
    noise_results: dict[float, dict[str, float]],
    metric: str = "accuracy",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """
    Plot model performance vs noise level.

    """
    # Use custom style if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    fig, ax = plt.subplots(figsize=figsize)

    # Extract data
    noise_levels = sorted(noise_results.keys())
    metric_values = [noise_results[level].get(metric, 0) for level in noise_levels]

    # Plot
    ax.plot(noise_levels, metric_values, marker="o", markersize=8, linewidth=2)
    ax.fill_between(noise_levels, metric_values, alpha=0.3)

    ax.set_xlabel("Noise Level (σ)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(
        f"Model Robustness to Noise\n{metric.replace('_', ' ').title()} vs Noise Level"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {save_path}")

    if show:
        plt.show()

    return fig
