import cupy as cp


def prepare_classification_batch(
    X_raw: cp.ndarray, y_raw: cp.ndarray, output_dim: int
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Standard preprocessing for classification tasks.
    Flattens input and one-hot encodes labels.
    """
    batch_size = X_raw.shape[0]

    # Flatten and transpose: (batch_size, ...) -> (input_dim, batch_size)
    X_batch = X_raw.reshape(batch_size, -1).T.astype(cp.float32)

    # One-hot encode labels: (output_dim, batch_size)
    y_onehot = cp.zeros((output_dim, batch_size), dtype=cp.float32)
    y_onehot[y_raw.astype(int), cp.arange(batch_size)] = 1.0

    return X_batch, y_onehot


def prepare_regression_batch(
    X_raw: cp.ndarray, y_raw: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Standard preprocessing for regression tasks.

    Flattens input and keeps targets as-is.
    """
    batch_size = X_raw.shape[0]

    # Flatten and transpose input
    X_batch = X_raw.reshape(batch_size, -1).T.astype(cp.float32)

    # Transpose targets (ensure 2D)
    if y_raw.ndim == 1:
        y_batch = y_raw.reshape(1, -1).astype(cp.float32)
    else:
        y_batch = y_raw.T.astype(cp.float32)

    return X_batch, y_batch


def mse_loss(y_true: cp.ndarray, y_pred: cp.ndarray) -> float:
    """
    Mean squared error loss for regression.
    """
    return float(cp.mean((y_true - y_pred) ** 2))


def mae_metric(y_true: cp.ndarray, y_pred: cp.ndarray) -> float:
    """
    Mean absolute error metric for regression.
    """
    return float(cp.mean(cp.abs(y_true - y_pred)))
