import numpy as np
import matplotlib.pyplot as plt


def examine_dataset(X_images, y_images):

    info = {
        "total_samples": X_images.shape[0],
        "image_shape": (28, 28),
        "num_classes": len(np.unique(y_images)),
        "unique_classes": np.unique(y_images),
        "class_distribution": np.bincount(y_images),
    }

    print(f"Total samples: {info['total_samples']}")
    print(f"Image shape: {info['image_shape']}")
    print(f"Number of classes: {info['num_classes']}")
    print(
        f"Classes range: {info['unique_classes'].min()} to {info['unique_classes'].max()}"
    )

    return info


def split_dataset(
    X_images, y_images, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
):
    """
    Split the dataset into train, validation, and test sets using stratified sampling.
    Args:
        X_images (np.array): Image data
        y_images (np.array): Label data
        train_size (float): Proportion of training data
        val_size (float): Proportion of validation data
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    np.random.seed(random_state)

    unique_classes = np.unique(y_images)

    train_indices = []
    val_indices = []
    test_indices = []

    for class_label in unique_classes:

        class_indices = np.where(y_images == class_label)[0]
        np.random.shuffle(class_indices)
        n_class_samples = len(class_indices)
        n_train = int(n_class_samples * train_size)
        n_val = int(n_class_samples * val_size)
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train : n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val :])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    X_train = X_images[train_indices]
    X_val = X_images[val_indices]
    X_test = X_images[test_indices]

    y_train = y_images[train_indices]
    y_val = y_images[val_indices]
    y_test = y_images[test_indices]

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_data(X_train, X_val, X_test):
    """
    Normalize the image data by dividing by 255
    """
    X_train_norm = X_train.astype("float32") / 255.0
    X_val_norm = X_val.astype("float32") / 255.0
    X_test_norm = X_test.astype("float32") / 255.0

    return X_train_norm, X_val_norm, X_test_norm


def flatten_images(X_train, X_val, X_test):
    """
    Flatten images from (n, 28, 28) to (n, 784) for neural network input.
    """
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    return X_train_flat, X_val_flat, X_test_flat


def preprocess_data(
    X_images, y_images, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
):
    """
    Complete preprocessing pipeline: split, normalize and flatten data.
    """

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X_images, y_images, train_size, val_size, test_size, random_state
    )

    X_train_norm, X_val_norm, X_test_norm = normalize_data(X_train, X_val, X_test)

    X_train_flat, X_val_flat, X_test_flat = flatten_images(
        X_train_norm, X_val_norm, X_test_norm
    )

    return X_train_flat, X_val_flat, X_test_flat, y_train, y_val, y_test
