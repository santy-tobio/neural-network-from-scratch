from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.typing import RcStyleType
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
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
    style: RcStyleType | None = None,
) -> Figure:
    """
    Visualize sample images from the dataset.

    """
    # Use custom style if available
    if style:
        plt.style.use(style)

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
    style: RcStyleType | None = None,
) -> Figure:
    """
    Plot a bar chart comparing multiple models on different metrics.

    """
    # Use custom style if available
    if style:
        plt.style.use(style)

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
    style: RcStyleType | None = None,
) -> Figure:
    """
    Plot model performance vs noise level.

    """
    # Use custom style if available
    if style:
        plt.style.use(style)

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


def plot_one_per_class(
    X,
    y,
    save_path="figures/one_per_class.png",
    figsize=(16, 12),
    show: bool = True,
    style: RcStyleType | None = None,
):
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


def plot_class_distribution(
    y,
    save_path="figures/class_distribution.png",
    figsize=(14, 6),
    show: bool = True,
    style: RcStyleType | None = None,
):
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
    _, ax = plt.subplots(figsize=figsize)

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


def plot_pixel_statistics(
    X,
    save_path="figures/pixel_statistics.png",
    figsize=(12, 5),
    show: bool = True,
    style: RcStyleType | None = None,
):
    """
    Plot mean and std of pixel values across the dataset.
    """
    import numpy as np
    from pathlib import Path

    # Use custom style if available
    if style:
        plt.style.use(style)
    plt.rcParams["figure.dpi"] = 300

    # Ensure figures directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Reshape if needed
    if X.ndim == 2 and X.shape[1] == 784:
        X_reshaped = X.reshape(-1, 28, 28)
    else:
        X_reshaped = X

    # Compute statistics
    mean_img = np.mean(X_reshaped, axis=0)
    std_img = np.std(X_reshaped, axis=0)

    # Create subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Mean
    im1 = ax1.imshow(mean_img, cmap="viridis")
    ax1.set_title("Mean Pixel Values", fontsize=12)
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Std
    im2 = ax2.imshow(std_img, cmap="viridis")
    ax2.set_title("Std Pixel Values", fontsize=12)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
