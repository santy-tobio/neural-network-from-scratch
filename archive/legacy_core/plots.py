import matplotlib.pyplot as plt
import os

plt.style.use("figs.mplstyle")
plt.rcParams["figure.dpi"] = 300


def visualize_sample_images(X_images, y_images, num_samples=3, figsize=(12, 4)):

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
